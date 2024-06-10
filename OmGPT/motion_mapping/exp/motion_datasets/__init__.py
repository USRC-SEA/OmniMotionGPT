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
provide dataloaders

'''
import os
from glob import glob

from motion_datasets.combine import CombineDataset
from motion_datasets.datasets import Text2MotionDataset
from motion_datasets.smal import Text2Motion
from torch.utils.data import DataLoader
from ylib.deforming_things_4d.prior import smal_animal_names_v1_train

SMPL_MAA = '../../_data/MDM/maa/'
SMPL_MAA_META = '../../_data/MDM/maa_meta/'
SMAL_MAA = '../../_data/DeformingThings4D/animals_maa_motions/'


def _process_list_filepath(list_filepath, process: bool):
    '''
    process the list of file path. this function is for quick debug propose 
    such that we do not need to load the entire dataset.
    '''
    if process:
        if len(list_filepath) > 16:
            list_filepath = list_filepath[:16]
    return list_filepath


def get_dataloader(args):
    '''
    get dataloader from args
    '''
    # SMPL
    # get the list of motion data
    list_smpl_maa = glob(os.path.join(SMPL_MAA, '*.npy'))
    list_smpl_maa = _process_list_filepath(
        list_smpl_maa, bool(args.data_debug)
    )
    t2m_dataset = Text2MotionDataset(
        list_filepath=list_smpl_maa
    )

    # SMAL
    list_smal_maa = glob(os.path.join(SMAL_MAA, 'bear3EP_*.npy'))
    list_smal_maa = _process_list_filepath(
        list_smal_maa, bool(args.data_debug)
    )
    smal_t2m_dataset = Text2Motion(list_filepath=list_smal_maa)

    # combine the dataset together
    combine_dataset = CombineDataset(
        smpl_t2m_dataset=t2m_dataset,
        smal_t2m_dataset=smal_t2m_dataset
    )

    loader = DataLoader(
        combine_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True, collate_fn=combine_dataset.collate
    )

    return loader


def get_smpl_dataloader(args):
    '''
    get dataloader for smpl motion and caption dataset.
    '''
    # SMPL
    # get the list of motion data
    list_smpl_maa = glob(os.path.join(SMPL_MAA, '*.npy'))
    list_smpl_maa = _process_list_filepath(
        list_smpl_maa, bool(args.data_debug)
    )
    t2m_dataset = Text2MotionDataset(
        list_filepath=list_smpl_maa
    )

    loader = DataLoader(
        t2m_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True, collate_fn=t2m_dataset.collate
    )

    return loader


def get_smal_dataloader(args):
    '''
    get dataloader for smal motion and caption dataset.
    '''
    # SMPL
    # get the list of motion data

    list_smal_maa = []
    for animal_name in smal_animal_names_v1_train:
        list_smal_maa.extend(
            glob(
                os.path.join(SMAL_MAA, f'{animal_name}_*.npy')
            )
        )

    list_smal_maa = _process_list_filepath(
        list_smal_maa, bool(args.data_debug)
    )
    smal_t2m_dataset = Text2Motion(list_filepath=list_smal_maa)

    loader = DataLoader(
        smal_t2m_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True, collate_fn=smal_t2m_dataset.collate
    )

    return loader
