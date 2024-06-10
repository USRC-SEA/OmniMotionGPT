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
visualize the results of the fitting process. the results are generated from
main.py in the same directory.

'''
import os

import cv2
from main import ANIMAL3D_TRAIN_JSON_FILE  # ==================================
from main import get_with_tail  # =============================================
from main import DT4D_OBJ_DIR, DT4D_SMAL_TEMPLATE_DIR
from tqdm import tqdm
from ylib.deforming_things_4d.prior import file_mapping
from ylib.obj import load_obj, save_obj
from ylib.obj.plot import plot_compare_obj, plot_obj

VISUALIZATION_DIR = '../../_runtime/data_process_00_fit'


def visualize_optimization_results(anime_name: str):
    '''
    visualize the optimization results of the given animal
    '''
    # load the DT4D mesh
    target_mesh_path = os.path.join(
        DT4D_OBJ_DIR, f'{anime_name}/00000.obj',
    )
    target_mesh = load_obj(target_mesh_path)

    # load the SMAL fitted template mesh
    smal_path = os.path.join(
        DT4D_SMAL_TEMPLATE_DIR, f'{anime_name}.obj',
    )
    smal_mesh = load_obj(smal_path)
    print(f'{anime_name:<25}', smal_mesh[0].shape, smal_mesh[1].shape)

    # to fix the reduandent vertice issue
    save_obj(smal_mesh[0], smal_mesh[1], smal_path)


def main():
    '''
    the main function for visualizing the results
    '''
    for anime_name in (file_mapping):
        visualize_optimization_results(anime_name)


if __name__ == '__main__':
    main()
