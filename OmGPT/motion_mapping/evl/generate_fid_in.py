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

import json
import os
import random
from glob import glob
from os.path import join as pjoin

import clip
import numpy as np
import torch
from eval import EvaluationDataset  # =========================================
from eval import _load_motion_mapping  # ======================================
from motion_datasets import get_dataloader  # =================================
from motion_datasets.utils import _lengths_to_mask  # =========================
from motion_models.architecture import MotionMapping  # =======================
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_activation_statistics  # =================
from utils.metrics import calculate_diversity  # ==============================
from utils.metrics import calculate_top_k  # ==================================
from utils.metrics import euclidean_distance_matrix  # ========================
from utils.options import parse_eval_args  # ==================================
from utils.utils import load_on_gpu  # ========================================
from utils.utils import unload_to_cpu  # ======================================
from ylib.print_options import options_str  # =================================
from ylib.torch import safe_l2_normalize  # ===================================
from ylib.torch.motion_datasets import get_smal_dataloader  # =================

MAA_OFFSET = '../../_data/DeformingThings4D/animals_maa_offsets/'
MAA = '../../_data/DeformingThings4D/animals_maa_motions/'


# def IDEvaluationDataset(EvaluationDataset):
#     '''
#     the dataset for evaluation for the in distribution
#     '''

#     def __init__(self):
#         '''
#         init function
#         '''
#         self.data = []

#         # get the list of maa files


# def _get_dataloader(args):
#     '''
#     create dataloader from the caption path and generated motion path
#     '''
#     # create dataset
#     dataset = IDEvaluationDataset()
#     # create data loader
#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=False,
#         num_workers=0,
#         drop_last=True,
#         collate_fn=dataset.collate
#     )

#     return loader


def main():
    '''
    main function for generate in distribution mean and cov
    '''
    # set args
    args, parser = parse_eval_args()
    print(options_str(args, parser))

    # load the trained model
    motion_mapping = MotionMapping(args)
    motion_mapping = nn.DataParallel(motion_mapping)
    _load_motion_mapping(motion_mapping, args)
    motion_mapping.to('cuda:0')
    motion_mapping.eval()

    # load the dataset
    # dataloader = _get_dataloader(args)
    args.data_debug = False
    args.batch_size = 32
    loader = get_smal_dataloader(args)

    # compute the mean and covariances
    list_latent = []
    with torch.no_grad():
        for data in tqdm(loader):
            batch_gpu_data = load_on_gpu(data)
            pred = motion_mapping.module.forward_from_smal(
                batch_gpu_data
            )
            list_latent.append(
                pred['motion_latent'].detach().cpu().numpy()
            )

    # reshape the latent
    latent = np.concatenate(list_latent, axis=0)
    latent = latent.reshape(latent.shape[0], -1)
    latent_mu, latent_cov = calculate_activation_statistics(latent)
    # save these two files in numpy
    np.save('latent_mu.npy', latent_mu)
    np.save('latent_cov.npy', latent_cov)


if __name__ == '__main__':
    main()
