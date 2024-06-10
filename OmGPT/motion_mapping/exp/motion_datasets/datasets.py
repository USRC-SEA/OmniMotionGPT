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
the dataset refer from mdm.data_loaders.humanml.data.dataset.
this code is very messy mainly due to humanml3d not our fault.

'''
import codecs as cs
import json
import os
import random
from os.path import join as pjoin
from typing import Dict, List

import clip
import numpy as np
import spacy
import torch
from data_loaders.humanml.utils.get_opt import get_opt
from data_loaders.humanml.utils.word_vectorizer import WordVectorizer
from motion_datasets.utils import collate_smpl
from torch.utils import data
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm


def _parse_line(line):
    """Parse a single line and return extracted details."""
    line_split = line.strip().split('#')
    caption = line_split[0]
    tokens = line_split[1].split(' ')

    # Handle potential NaN values before converting to float
    if np.isnan(float(line_split[2])):
        f_tag = 0.0
    else:
        f_tag = float(line_split[2])

    if np.isnan(float(line_split[3])):
        to_tag = 0.0
    else:
        to_tag = float(line_split[3])

    return caption, tokens, f_tag, to_tag


def _generate_unique_name(base_name: str, existing_data) -> str:
    """Generate a unique name by adding a random prefix to the base name."""
    new_name = random.choice(
        'ABCDEFGHIJKLMNOPQRSTUVW'
    ) + '_' + base_name
    while new_name in existing_data:
        new_name = random.choice(
            'ABCDEFGHIJKLMNOPQRSTUVW'
        ) + '_' + base_name
    return new_name


def _is_valid_motion_length(motion, min_len, max_len):
    """Check if the motion's length is within the valid range."""
    return min_len <= len(motion) < max_len


class Text2MotionDataset(data.Dataset):
    '''
    Dataset for text to motion
    '''

    def _get_text_path(self, maa_path: str) -> str:
        '''
        get the text path from the maa path
        '''
        split_path = maa_path.split('/')

        # replace folder path
        split_path[-2] = 'texts'

        # replace file extension
        maa_filename = split_path[-1]
        split_path[-1] = maa_filename.split('.')[0] + '.txt'

        return '/'.join(split_path)

    def _handle_motion(
        self, f_tag, to_tag, motion, text_entry, name,
    ):
        """
        Process motion data and update the data structures.
        """
        n_motion = motion[int(f_tag * 20): int(to_tag * 20)]

        if not _is_valid_motion_length(
            motion, self.min_motion_len, 200
        ):
            return

        unique_name = _generate_unique_name(name, self.data_dict)
        self.data_dict[unique_name] = {
            'motion': n_motion,
            'length': len(n_motion),
            'text': [text_entry]
        }
        self.new_name_list.append(unique_name)
        self.length_list.append(len(n_motion))

    def _parse_text_file(
        self, text_file: str, motion: np.ndarray,
    ) -> None:

        text_data = []
        name = text_file.split('/')[-1].split('.')[0]

        # get the valid subjects
        subjects = self.subjects[text_file]
        valid_subjects = []
        for subject in subjects:
            if len(subject) > 0:
                valid_subjects.append(subject[0])
            else:
                valid_subjects.append('')

        open_file = cs.open(text_file)

        for i, line in enumerate(open_file.readlines()):
            caption, tokens, f_tag, to_tag = _parse_line(line)

            text_entry = {
                'caption': caption, 'tokens': tokens,
                'subject': valid_subjects[i],
            }

            if f_tag == 0.0 and to_tag == 0.0:
                text_data.append(text_entry)
                continue

            try:
                self._handle_motion(
                    f_tag, to_tag, motion, text_entry, name
                )
            except Exception as exception:
                print(f"Error processing motion: {exception}")
                print(line)
                print(f_tag, to_tag, name)

        # close the file
        open_file.close()

        if len(text_data) > 0:

            self.data_dict[name] = {
                'motion': motion,
                'length': len(motion),
                'text': text_data,
                # 'subject': valid_subjects,
            }
            self.new_name_list.append(name)
            self.length_list.append(len(motion))

    def __init__(self, list_filepath):
     

        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = 196
        self.min_motion_len = 40
        self.unit_length = 4

        self.data_dict = {}
     

        # load the subjects
        with open('../../_data/MDM/subjects.json', 'r') as f:
            self.subjects = json.load(f)

        self.new_name_list = []
        self.length_list = []

        for filepath in tqdm(list_filepath):


            # load motion
            motion = np.load(filepath)

            if not _is_valid_motion_length(
                motion, self.min_motion_len, 200
            ):
                continue

            # load text data
            text_path = self._get_text_path(filepath)
            self._parse_text_file(text_path, motion)

        name_list, length_list = zip(
            *sorted(
                zip(self.new_name_list, self.length_list),
                key=lambda x: x[1]
            )
        )

        self.mean = np.load('../../_data/MDM/maa_meta/mean.npy')
        self.std = np.load('../../_data/MDM/maa_meta/std.npy')
        self.offset = np.load('../../_data/MDM/maa_meta/offset.npy')
        self.length_arr = np.array(length_list)
        # self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)
        self.max_text_len = 20

    def _load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(
            clip_version, device='cpu',
            jit=False
        )  

        clip.model.convert_weights( 
            clip_model
        )  

        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False

        return clip_model

    @staticmethod
    def _tokenize_text(raw_text):
        '''
        encode text with clip.

        inputs:
        -------
        raw_text
            list (batch_size length) of strings with input text prompts

        return:
        -------
        tensor : [batch_size, 512]
            the clip text feature
        '''
        # Specific hardcoding for humanml dataset
        max_text_len = 20

        default_context_length = 77
        context_length = max_text_len + 2  # start_token + 20 + end_token
        assert context_length < default_context_length

        # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        )

        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype
        )
        texts = torch.cat([texts, zero_pad], dim=1)
        return texts

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):

        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):

        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        subject = text_data['subject']

        if len(tokens) < self.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (
                self.max_text_len + 2 - sent_len
            )
        else:
            # crop
            tokens = tokens[:self.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        num_chunck = m_length // self.unit_length
        if coin2 == 'double':
            m_length = (num_chunck - 1) * self.unit_length
        elif coin2 == 'single':
            m_length = num_chunck * self.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]


        masked_std = self.std.copy()
        masked_std[masked_std == 0] = 1
        motion = (motion - self.mean) / masked_std

        if m_length < self.max_motion_length:
            motion = np.concatenate(
                [
                    motion,
                    np.zeros(
                        (
                            self.max_motion_length - m_length,
                            motion.shape[1],
                            motion.shape[2],
                        )
                    )
                ],
                axis=0
            )

        return caption, sent_len, motion, m_length, '_'.join(tokens), self.offset, \
            self.mean, self.std, self.name_list[idx], subject

    @classmethod
    def collate(cls, batch):
        '''
        the collate function for the human motion dataset.
        '''
        smpl_meta = {
            'mean': [b[6] for b in batch],
            'std':  [b[7] for b in batch],
            'name': [b[8] for b in batch],
        }

        real_smpl_input, _ = collate_smpl(batch)

        # tokenlize the text data
        real_smpl_input['token_text'] = cls._tokenize_text(
            real_smpl_input['smpl_text']
        )

        return {
            'real_smpl_input': real_smpl_input,
            'smpl_meta': smpl_meta,
        }
