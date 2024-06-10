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
evaluate in-distribution multi-modality performance of generated motion
'''
import json
import os
import random
from glob import glob
from os.path import join as pjoin

import clip
import numpy as np
import torch
from eval import _get_list_caption_path  # ====================================
from eval import _load_motion_mapping  # ======================================
from eval import _sample_two_list  # ==========================================
from eval_mm import _check_generated_motion_number  # =========================
from eval_mm import _eval_one_epoch  # ========================================
from motion_datasets import get_dataloader  # =================================
from motion_datasets.utils import _lengths_to_mask  # =========================
from motion_models.architecture import MotionMapping  # =======================
from scipy import linalg
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


class EvaluationDataset(data.Dataset):
    '''
    evaluation dataset for mm motion generated with the same caption
    '''

    def _load_caption(self, caption_file_path):
        '''
        load caption from the caption file path.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        caption = meta['smal_text']
        return caption

    def _load_motion_length(self, caption_file_path):
        '''
        load motion length from the caption file path.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        motion_length = meta['motion_length']
        return motion_length

    def _load_gmotion(self, gmotion_dir):
        '''
        load generated motion from the generated motion file path.
        '''
        list_motion_path = glob(pjoin(gmotion_dir, 'motion*.npy'))
        list_motion_path.sort()
        assert len(list_motion_path) >= 10, 'at least 10 motions.'

        list_motion = []
        for motion_path in list_motion_path:
            motion = np.load(motion_path)
            list_motion.append(motion)
        motion = np.stack(list_motion, axis=0)
        # only keep the first 10 motions
        motion = motion[:10, ...]

        return motion

    def _process_motion_to_maa(self, raw_motion):
        '''
        the current motion is represented as 321 and the motion should be 36*6
        '''
        if len(raw_motion.shape) == 4:
            return raw_motion
        assert len(raw_motion.shape) == 3, 'the motion should be 3d.'
        maa_len = 36 * 6
        motion = raw_motion[..., :maa_len]
        if motion.shape[1] < 196:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        [motion.shape[0], 196 - motion.shape[1], 36 * 6]
                    )
                ],
                axis=1
            )

        motion = motion.reshape(-1, 196, 36, 6)
        return motion

    def _load_animal_id(self, caption_file_path):
        '''
        load animal id from meta file.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        animal_id = meta['smal_id'].split('/')[-1].split('_')[0]
        return animal_id

    def _load_offset(self, animal_id):
        '''
        load offset from animal id
        '''
        offset_path = pjoin(MAA_OFFSET, f'{animal_id}', 'offset.pkl')
        offset = np.load(offset_path, allow_pickle=True)
        return offset

    def __init__(self, caption_path, gmotion_dir) -> None:
        '''
        gmotion is short for generated motion
        '''
        self.data = []
        for caption_file_path, gmotion_dir in tqdm(
            zip(caption_path, gmotion_dir), desc='loading data',
        ):
            caption = self._load_caption(caption_file_path)
            gmotion = self._load_gmotion(gmotion_dir)
            motion_length = self._load_motion_length(caption_file_path)
            maa_motion = self._process_motion_to_maa(gmotion)
            animal_id = self._load_animal_id(caption_file_path)
            offset = self._load_offset(animal_id)

            # repeat motion_length 10 times
            motion_length = np.repeat(motion_length, 10)
            # repeat offset 10 times in new axis
            offset = np.repeat(offset[np.newaxis, :], 10, axis=0)

            self.data.append(
                {
                    'caption': caption,
                    'motion': maa_motion,
                    'motion_length': motion_length,
                    'offset': offset,
                }
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        get item from the dataset
        '''
        return self.data[index]

    @classmethod
    def collate(cls, batch):
        '''
        the collate function for the evaluation motion dataset
        '''
        # list_text = [item['caption'][0] for item in batch]
        # clip_token = cls._tokenize_text(list_text)
        # list_motions = [torch.from_numpy(item['motion']) for item in batch]
        motions = torch.from_numpy(batch[0]['motion']).float()
        # motions = torch.stack(list_motions, dim=0)

        # mask
        # lengths = torch.tensor([item['motion_length'] for item in batch])
        lengths = torch.from_numpy(batch[0]['motion_length'])
        mask = _lengths_to_mask(lengths, motions.shape[1])

        # offset
        # list_offsets = [torch.from_numpy(item['offset']) for item in batch]
        # offsets = torch.stack(list_offsets, dim=0)
        offsets = torch.from_numpy(batch[0]['offset'])

        return {
            # 'clip_token': clip_token,
            'motions': motions,
            'offsets': offsets,
            'mask': mask,
        }


def _create_dataloader(sampled_caption_path, smapled_generated_motion_dir):
    '''
    create dataloader only for mm evaluation.
    '''
    # create dataset
    dataset = EvaluationDataset(
        sampled_caption_path, smapled_generated_motion_dir
    )
    # create data loader
    loader = DataLoader(
        dataset,
        batch_size=1,  # batch size is 1 but load 10 motions a time
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=dataset.collate
    )

    return loader


def main():
    '''
    the main function for evaluate the mm for ood of model
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

    # we do evaluation for 20 times
    num_eval = 20
    list_mm = []
    # list_rp = []
    # list_dv = []
    for current_eval_i in range(20):
        sampled_caption_path, smapled_generated_motion_path = _sample_two_list(
            list_caption_path, list_generated_motion_path, num_eval
        )

        # create dataloader from the names
        loader = _create_dataloader(
            sampled_caption_path, smapled_generated_motion_path
        )

        multi_modality = _eval_one_epoch(loader, motion_mapping)
        list_mm.append(multi_modality)
        # list_rp.append(rp)
        # list_dv.append(dv)

    # compute the mean
    mm_mean = np.mean(list_mm)
    mm_std = np.std(list_mm)
    print(f'multi modality mean: {mm_mean:.3f} std: {mm_std:.3f}')


if __name__ == '__main__':
    main()
