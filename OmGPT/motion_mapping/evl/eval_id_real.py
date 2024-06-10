# MIT License

# Copyright (c) 2024 OPPO

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
'''
evaluate in-distribution performance of real motions and compute metrics

'''
import json
import os
import random
from glob import glob
from os.path import join as pjoin

import clip
import numpy as np
import torch
from eval import _create_dataloader  # ========================================
from eval import _eval_one_epoch  # ===========================================
from eval import _load_motion_mapping  # ======================================
from eval import _sample_two_list  # ==========================================
from eval_fid_out import _eval_fid_out_one_epoch  # ===========================
from motion_datasets import get_dataloader  # =================================
from motion_datasets.utils import _lengths_to_mask  # =========================
from motion_models.architecture import MotionMapping  # =======================
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_diversity  # ==============================
from utils.metrics import calculate_top_k  # ==================================
from utils.metrics import euclidean_distance_matrix  # ========================
from utils.options import parse_eval_args  # ==================================
from utils.utils import load_on_gpu  # ========================================
from utils.utils import unload_to_cpu  # ======================================
from ylib.print_options import options_str  # =================================
from ylib.torch import safe_l2_normalize  # ===================================

MAA_OFFSET = '../../_data/DeformingThings4D/animals_maa_offsets/'


def _check_generated_motion_number(args):
    '''
    check that the generated motion has the same number as the caption
    '''
    # gather two list
    list_caption_path = glob(pjoin(args.caption_path, '*/meta.json'))
    list_caption_path.sort()

    list_generated_motion_path = glob(
        pjoin(args.generated_motion_dir, '*.npy')
    )
    list_generated_motion_path.sort()

    # assert len(list_caption_path) == len(list_generated_motion_path)

    # match the order
    list_motion_id = [
        caption_path.split('/')[-2] for caption_path in list_caption_path
    ]
    new_list_generated_motion_path = []
    new_list_caption_path = []
    for generated_motion_path in list_generated_motion_path:
        generated_motion_name = generated_motion_path.split(
            '/')[-1].split('.')[0]
        if generated_motion_name in list_motion_id:
            new_list_generated_motion_path.append(generated_motion_path)
            new_list_caption_path.append(
                list_caption_path[list_motion_id.index(generated_motion_name)]
            )

    # double check correct correspondence
    for caption_path, generated_motion_path in zip(
        new_list_caption_path, new_list_generated_motion_path
    ):
        caption_name = caption_path.split('/')[-2]
        generated_motion_name = generated_motion_path.split(
            '/')[-1].split('.')[0]
        assert caption_name == generated_motion_name, (
            f'{caption_name} != {generated_motion_name}'
        )
    print('number of generated motion: ', len(new_list_generated_motion_path))
    print('number of caption: ', len(new_list_caption_path))

    return new_list_caption_path, new_list_generated_motion_path


def _save_one_epoch(
    motion_mapping, list_caption_path, list_generated_motion_path
):
    '''
    save the activation for one epoch.
    '''
    loader = _create_dataloader(
        list_caption_path, list_generated_motion_path
    )
    list_pred = []
    with torch.no_grad():
        for data in loader:
            batch_gpu_data = load_on_gpu(data)
            pred = motion_mapping.module.forward_for_evaluation(
                batch_gpu_data
            )
            list_pred.append(pred)

    # save text emd and text emd_pred
    text_emd = torch.cat([pred['text_emd'] for pred in list_pred], dim=0)
    motion_latent = torch.cat(
        [pred['motion_latent'] for pred in list_pred], dim=0
    )
    torch.save(text_emd, 'tmp/gt/text_emd.pt')
    torch.save(motion_latent, 'tmp/gt/motion_latent.pt')


def main():
    '''
    the main function for load the motion, model, and generate metric results
    '''
    # set args
    args, parser = parse_eval_args()
    print(options_str(args, parser))

    # check that the caption folder's name matches the generated motion
    # folder's names
    list_caption_path, list_generated_motion_path = _check_generated_motion_number(
        args
    )

    # load the trained model
    motion_mapping = MotionMapping(args)
    motion_mapping = nn.DataParallel(motion_mapping)
    _load_motion_mapping(motion_mapping, args)
    motion_mapping.to('cuda:0')
    motion_mapping.eval()

    # save the avtivations and text features
    _save_one_epoch(
        motion_mapping, list_caption_path, list_generated_motion_path
    )

    # we do evaluation for 20 times
    num_eval = 64
    list_ms = []
    list_rp = []
    list_dv = []
    list_fid = []
    for current_eval_i in range(20):
        sampled_caption_path, smapled_generated_motion_path = _sample_two_list(
            list_caption_path, list_generated_motion_path, num_eval
        )

        # create dataloader from the names
        loader = _create_dataloader(
            sampled_caption_path, smapled_generated_motion_path
        )

        ms, rp, dv = _eval_one_epoch(loader, motion_mapping)
        fid = _eval_fid_out_one_epoch(loader, motion_mapping)

        list_ms.append(ms)
        list_rp.append(rp)
        list_dv.append(dv)
        list_fid.append(fid)

    # compute the mean
    ms_mean = np.mean(list_ms)
    ms_std = np.std(list_ms)
    rp_mean = np.mean(list_rp, axis=0)
    rp_std = np.std(list_rp, axis=0)
    dv_mean = np.mean(list_dv)
    dv_std = np.std(list_dv)
    print(
        f'matching score (mm-dist) mean: {ms_mean:.4f} std: {ms_std:.4f}'
    )

    # Create a formatted string for each top-k result
    formatted_results = [
        f'{mean:.3f}\u00B1{std:.3f}' for mean, std in zip(rp_mean, rp_std)
    ]

    # Join the formatted results into a single string separated by spaces
    formatted_string = ' '.join(formatted_results)

    print(f'r precision mean: {formatted_string}')

    print(f'diversity mean: {dv_mean:.3f} std: {dv_std:.3f}')

    # compute the mean
    fid_mean = np.mean(list_fid)
    fid_std = np.std(list_fid)
    print(f'fid mean: {fid_mean:.4f} std: {fid_std:.4f}')


if __name__ == '__main__':
    main()
