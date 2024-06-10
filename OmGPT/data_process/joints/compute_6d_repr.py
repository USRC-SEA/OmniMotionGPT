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
import pickle
import subprocess
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import List

import numpy as np
import torch
from ylib.deforming_things_4d.prior import file_mapping  # ====================
from ylib.deforming_things_4d.prior import smal_animal_names_v1  # ============
from ylib.maa import convert_motion_to_maa  # =================================
from ylib.skeleton_prior import smal_35_face_joint_indx  # ====================
from ylib.skeleton_prior import smal_35_kinematic_chain  # ====================
from ylib.skeleton_prior import smal_35_parents  # ============================
from ylib.skeleton_prior import smal_symmetry_joint_index  # ==================
from ylib.smpl_format.pre_processing import preprocess_joints  # ==============

DT4D_TEMPLATE_DIR = '../../_data/DeformingThings4D/animals_smal_template/'
DT4D_SMAL_JOINTS_DIR = '../../_data/DeformingThings4D/animals_smal_joints/'
DT4D_MAA_OFFSETS_DIR = '../../_data/DeformingThings4D/animals_maa_offsets/'
DT4D_VIS_DIR = '../../_data/DeformingThings4D/animals_vis/'
DT4D_MAA_MOTIONS_DIR = '../../_data/DeformingThings4D/animals_maa_motions/'


def load_scale(animal_name) -> float:
    '''
    load scale from the template
    '''
    json_file = glob(f'{DT4D_TEMPLATE_DIR}/{animal_name}_*.json')

    # make sure that there is only one json file
    assert len(json_file) == 1

    with open(json_file[0], 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    scale = json_data['scale'][0][0][0]

    return scale


def get_motion_name(animal_name: str) -> str:
    '''
    get the motion name from the animal name
    '''

    for key in file_mapping:
        if animal_name in key:
            return key

    raise ValueError(f'animal name {animal_name} not found')


def compute_offset_from_joints(
    joints: np.ndarray, parents: List[int]
) -> np.ndarray:
    '''
    compute the offset from the joints

    inputs:
    -------
    joints: np.ndarray
        the joints position, shape: (num_joints, 3)
    '''
    offset = np.zeros_like(joints)
    for i, parent in enumerate(parents):
        if parent == -1:
            continue
        offset[i] = joints[i] - joints[parent]
    return offset


def process_joint_to_6d(
    animal_name: str, offset: np.ndarray, scale: float
) -> None:
    '''
    process the joints to 6d representation
    '''
    # get the motion name
    list_motion_file = glob(
        os.path.join(DT4D_SMAL_JOINTS_DIR, f'{animal_name}_*.npy')
    )

    for motion_file in list_motion_file:
        # load the joints
        joints = np.load(motion_file)

        # for joints, we need to scale the joints
        joints = joints / scale

        # pre-process the joints
        joints = preprocess_joints(joints, smal_35_face_joint_indx)

        # compute the 6d representation
        maa_motions = convert_motion_to_maa(
            positions=torch.tensor(joints).float(),
            offset=offset,
            kinematic_chain=smal_35_kinematic_chain,
            face_joint_index=smal_35_face_joint_indx,
        )

        # save motions
        motion_name = motion_file.split('/')[-1].split('.')[0]
        maa_motion_file = os.path.join(
            DT4D_MAA_MOTIONS_DIR, f'{motion_name}.npy'
        )
        os.makedirs(os.path.dirname(maa_motion_file), exist_ok=True)
        np.save(maa_motion_file, maa_motions)
        print(f'saved {maa_motion_file}!')


def process(animal_name):
    '''
    the process function for parallel processing to process the animal motions.
    '''
    # load scale
    scale = load_scale(animal_name)

    # get the joints file position
    motion_name = get_motion_name(animal_name)
    joint_file_path = os.path.join(
        DT4D_SMAL_JOINTS_DIR, f'{motion_name}.npy'
    )

    # load the joints position
    joints = np.load(joint_file_path)

    # for joints, we need to scale the joints
    joints = joints / scale

    # get the first frame
    first_frame = joints[0]

    # make the joints symmetric around x axis
    mirror_joints = first_frame.copy()
    mirror_joints[:, 0] = -mirror_joints[:, 0]
    mirror_joints = mirror_joints[smal_symmetry_joint_index, :]
    symmetric_joints = (first_frame + mirror_joints) / 2.

    # compute the offset
    offset = compute_offset_from_joints(symmetric_joints, smal_35_parents)

    # save the offset
    offset_file_path = os.path.join(
        DT4D_MAA_OFFSETS_DIR, f'{animal_name}', 'offset.pkl'
    )
    os.makedirs(os.path.dirname(offset_file_path), exist_ok=True)
    with open(offset_file_path, 'wb') as open_file:
        pickle.dump(offset, open_file)

    # process each character
    process_joint_to_6d(animal_name, offset, scale)


def main():
    '''
    main function for procssing skeleton data
    '''
    num_threads = os.cpu_count()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process, smal_animal_names_v1))
    print('done', results)
 

if __name__ == '__main__':
    main()
