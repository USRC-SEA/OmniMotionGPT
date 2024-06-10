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

import os
from glob import glob
from typing import List

import numpy as np
import torch
from tqdm import tqdm
from ylib.obj import load_obj
from ylib.skeleton_obj_writer import draw_skeleton_animation  # ===============
from ylib.skeleton_prior import smal_35_parents  # ============================
from ylib.smal import SMAL_FACE_NUM as smal_face_num  # =======================
from ylib.smal import SMAL_NO_TAIL_FACE_NUM as smal_no_tail_face_num  # =======
from ylib.smal import init_smal_model  # ======================================
from ylib.smal import no_tail_vert_to_smal_vert  # ============================

DT4D_SMAL_JOINTS_DIR = '../../_data/DeformingThings4D/animals_smal_joints'
DT4D_SMAL_OBJ_DIR = '../../_data/DeformingThings4D/animals_smal_obj'


def gather_list_vertices(obj_file_paths: List[str]):
    '''
    gather the vertices from the obj file paths
    '''
    list_vertices = []
    for obj_file_path in obj_file_paths:
        # load obj file
        loaded_mesh = load_obj(obj_file_path)
        # look into the number faces
        if loaded_mesh[1].shape[0] == smal_face_num:
            # the smal model with tail
            vertices = loaded_mesh[0]
        elif loaded_mesh[1].shape[0] == smal_no_tail_face_num:
            # the smal model without tail, process the vertices
            vertices = no_tail_vert_to_smal_vert(loaded_mesh[0])
        else:
            raise ValueError('the number of faces is not correct')
        list_vertices.append(vertices)

    return np.stack(list_vertices, axis=0)


def process_motion_dir(motion_dir, smal_model):
    '''
    process the SMAL obj file
    '''
    # get the obj file path
    obj_file_paths = glob(
        os.path.join(motion_dir, '*.obj')
    )

    vertices = gather_list_vertices(obj_file_paths)

    # compute the joints from vertices
    joints = smal_model._regress_joint(
        torch.from_numpy(vertices).float().to(device=smal_model.device)
    )

    # we need to rotate the vertices back
    align_rotation = torch.tensor(
        [
            [1, 0, 0],
            [0,  0, 1],
            [0,  -1, 0],
        ],
        dtype=torch.float32,
        device=smal_model.device,
    )
    joints = joints @ align_rotation.T.unsqueeze(0)

    # save the joints
    save_path = os.path.join(
        DT4D_SMAL_JOINTS_DIR, motion_dir.split('/')[-1] + '.npy'
    )
    np.save(save_path, joints.detach().data.cpu().numpy())


def main():
    '''
    the main function to convert vertices to joints
    '''
    # get all the obj file path
    motion_dir_paths = glob(
        os.path.join(DT4D_SMAL_OBJ_DIR, '*')
    )

    # create smal model
    _, smal_model = init_smal_model([0.0] * 20, torch.device('cuda:0'))

    # process the obj file path
    for motion_dir_path in tqdm(motion_dir_paths):
        process_motion_dir(motion_dir_path, smal_model)


if __name__ == '__main__':
    main()
