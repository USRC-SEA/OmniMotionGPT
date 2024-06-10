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
the main function for fit a initial smal model onto target mesh.

'''
import json
import os
from glob import glob
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
from ylib.deforming_things_4d.prior import file_mapping
from ylib.obj import load_obj, save_obj
from ylib.skeleton_obj_writer import get_limit_from_joints  # ================
from ylib.skeleton_obj_writer import set_axes_equal  # ========================
from ylib.smal.smal_model import SMALModel

DT4D_OBJ_DIR = '../../_data/DeformingThings4D/animals_obj/'
ANIMAL3D_TRAIN_JSON_FILE = '../../_data/Animal3D/animal3d/train.json'
DT4D_SMAL_TEMPLATE_DIR = '../../_data/DeformingThings4D/animals_smal_template'


class SMALConfig:
    '''
    the configuration of smal model.
    '''

    def __init__(
        self,
        beta: torch.Tensor,
        betas_logscale: torch.Tensor,
        scale: torch.Tensor,
        smal_faces: np.ndarray,
        theta: torch.Tensor,
        trans: torch.Tensor,
    ):
        self._beta = beta
        self._betas_logscale = betas_logscale
        self._scale = scale
        self._theta = theta
        self._smal_faces = smal_faces
        self._trans = trans

    @property
    def beta(self) -> torch.Tensor:
        '''
        return the beta tensor
        '''
        return self._beta

    @property
    def betas_logscale(self) -> torch.Tensor:
        '''
        return the betas_logscale tensor
        '''
        return self._betas_logscale

    @property
    def theta(self) -> torch.Tensor:
        '''
        return the theta tensor
        '''
        return self._theta

    @property
    def scale(self) -> torch.Tensor:
        '''
        return the scale tensor
        '''
        return self._scale

    @property
    def smal_faces(self) -> np.ndarray:
        '''
        return the smal faces
        '''
        return self._smal_faces

    @property
    def trans(self) -> torch.Tensor:
        '''
        global translation
        '''
        return self._trans

    def to_json(self):
        '''
        convert the internal data to json writable format
        '''
        return {
            'beta': self._beta.detach().data.cpu().numpy().tolist(),
            'betas_logscale': self._betas_logscale.detach().data.cpu().numpy().tolist(),
            'theta': self._theta.detach().data.cpu().numpy().tolist(),
            'scale': self._scale.detach().data.cpu().numpy().tolist(),
            'trans': self._trans.detach().data.cpu().numpy().tolist(),
        }


def plot_obj(vertices, faces):
    '''
    plot the obj file
    '''
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # for face in tqdm(faces, desc='plotting'):
    #     vertex1 = vertices[face[0]]
    #     vertex2 = vertices[face[1]]
    #     vertex3 = vertices[face[2]]

    #     x = [vertex1[0], vertex2[0], vertex3[0], vertex1[0]]
    #     y = [vertex1[1], vertex2[1], vertex3[1], vertex1[1]]
    #     z = [vertex1[2], vertex2[2], vertex3[2], vertex1[2]]

    #     ax.plot(x, y, z, c='b')
    polygons = [[vertices[face[j]] for j in range(3)] for face in faces]
    poly3d = Poly3DCollection(
        polygons, edgecolor='k', facecolor='cyan', alpha=0.25,
        linewidth=0.5
    )

    ax.add_collection3d(poly3d)

    # ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    # ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    # ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    x_limit, y_limit, z_limit = get_limit_from_joints(vertices)

    set_axes_equal(x_limit, y_limit, z_limit, ax)
    # ax.margins(0)
    plt.tight_layout()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8  # type: ignore
    )

    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False  # Transparent panes

    # clean the figure
    plt.close()

    return image


def filter_tail(vertices, no_tail_vertices, faces):
    # This is a placeholder. You'll need to determine the actual range
    # of vertices that correspond to the tail.
    # if vertices.dtype != np.float32:
    vertices = vertices.astype(np.float32)
    quat_vertices = vertices * 50
    quat_vertices = quat_vertices.astype(np.int32)
    # if no_tail_vertices.dtype != np.float32:
    no_tail_vertices = no_tail_vertices.astype(np.float32)
    quat_no_tail_vertices = no_tail_vertices * 50
    quat_no_tail_vertices = quat_no_tail_vertices.astype(np.int32)

    tail_vertices_indices = []
    for vertex_i, vertex in enumerate(quat_vertices):
        if not is_vertex_in_array(vertex, quat_no_tail_vertices):
            tail_vertices_indices.append(vertex_i)
    # tail_vertices_indices = set(range(START_INDEX, END_INDEX))

    # Filter out faces that have any vertex in the tail
    filtered_faces = [
        face for face in faces
        if not any(v in tail_vertices_indices for v in face)
    ]
    filtered_faces = np.stack(filtered_faces)

    return vertices, filtered_faces


def is_vertex_in_array(query_vertex, vertices):
    """
    Determine if a given vertex is in an n x 3 numpy array of vertices.

    Parameters:
    - query_vertex (numpy array): Array representing the vertex to be checked, of shape (1, 3).
    - vertices (numpy array): Array of shape (n, 3) representing n vertices.

    Returns:
    - bool: True if the vertex is in the array, False otherwise.
    """

    return np.any(np.all(vertices == query_vertex, axis=1))


def get_beta_from_json(json_file: str, img_keyword: str) -> List[float]:
    '''
    load beta from json file with the image path keyword.
    '''
    beta = []
    with open(json_file, 'r', encoding='utf-8') as f:
        train_meta = json.load(f)

    for data_info in train_meta['data']:
        if img_keyword in data_info['img_path']:
            beta = data_info['shape']  # 20
            break

    assert len(beta) == 20, 'beta should be 20'

    return beta


def get_with_tail(json_file: str, img_keyword: str) -> bool:
    '''
    get with_tail information from json file with img_keyword.
    '''
    with_tail = None
    with open(json_file, 'r', encoding='utf-8') as f:
        train_meta = json.load(f)

    for data_info in train_meta['data']:
        if img_keyword in data_info['img_path']:
            with_tail = bool(data_info['with_tail'])
            break

    if with_tail is None:
        raise ValueError('with_tail is None')

    return with_tail


def init_smal_model(
    list_beta: List[float], device
) -> Tuple[nn.Parameter, SMALModel]:
    '''
    initialize the smal model with an initial guess of beta.
    '''
    # this is just for initialize SMALModel
    beta_np = np.array(list_beta, dtype=np.float32)

    # this is for forward function
    beta = torch.tensor(beta_np).to(device).float()
    beta = torch.nn.Parameter(beta, requires_grad=True)

    # for initiazlie SMALModel
    betas_logscale_np = np.zeros((1, 6), dtype=np.float32)

    smal_model = SMALModel(
        beta_np, device, betas_logscale_np,
    )

    return beta,  smal_model


def get_poly3d(vertices, faces, face_color='cyan'):
    '''
    get the poly3d object from vertices and faces.
    '''
    polygons = [[vertices[face[j]] for j in range(3)] for face in faces]
    poly3d = Poly3DCollection(
        polygons, edgecolor='k', facecolor=face_color, alpha=0.25,
        linewidth=0.5
    )
    return poly3d


def plot_compare_obj(vertices_0, faces_0, vertices_1, faces_1):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    poly3d_0 = get_poly3d(vertices_0, faces_0, face_color='cyan')
    ax.add_collection3d(poly3d_0)

    poly3d_1 = get_poly3d(vertices_1, faces_1, face_color='r')
    ax.add_collection3d(poly3d_1)

    x_limit, y_limit, z_limit = get_limit_from_joints(vertices_0)

    set_axes_equal(x_limit, y_limit, z_limit, ax)
    plt.tight_layout()
    ax.grid(True, linestyle='--', alpha=0.6)

    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_rgb(), dtype=np.uint8  # type: ignore
    )

    # Reshape the data into an image
    image = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False  # Transparent panes

    # clean the figure
    plt.close()

    return image


def smal_vert_to_no_tail_vert(
    smal_vertices, smal_no_tail_index
):
    '''
    convert the vertices in smal model to no tail smal model.
    '''
    pass


def no_tail_vert_to_smal_vert(
    no_tail_vertices, smal_no_tail_index
):
    '''
    take some time to figure out the index. 
    '''
    tail_index = 449

    smal_vertices = np.zeros((3889, 3), dtype=np.float32)
    smal_vertices[:] = no_tail_vertices[tail_index]
    smal_vertices[smal_no_tail_index] = no_tail_vertices

    return smal_vertices


def generate_no_tail_index(ordered_no_tail_faces):
    '''
    generate the index of no tail vertices.
    '''
    vertices = np.arange(3889)
    # expand vertices to 3 times
    # vertices : [3889, 3]
    vertices = torch.from_numpy(vertices)[:, None].repeat(1, 3)
    save_obj(vertices, ordered_no_tail_faces, './tmp_good.obj')
    # index : [3800, 3]
    index, _ = load_obj('./tmp_good.obj')
    index = torch.from_numpy(index).long()
    index = index[:, 0].numpy().tolist()
    with open('smal_no_tail_index.json', 'w', encoding='utf-8') as f:
        json.dump(index, f)
    return index


def optimize_smal_model(
    num_epoch: int,
    optimizer: torch.optim.Optimizer,
    smal_config: SMALConfig,
    smal_model: SMALModel,
    target_mesh_tuple: Tuple[np.ndarray, np.ndarray],
    visualize_interval: int = -1,
):
    '''
    optimize the smal model to fit the target mesh.

    inputs:
    -------
    num_epoch: int

    '''
    device = smal_model.device
    align_rotation = torch.tensor(
        [
            [0,   1, 0],
            [-1,  0, 0],
            [0,   0, 1],
        ],
        dtype=torch.float32,
        device=smal_model.device,
    )
    for epoch in tqdm(range(num_epoch)):
        # verts : [batch_size, 3889, 3]
        verts, joints, Rs, v_shaped = smal_model(
            beta=smal_config.beta,
            theta=smal_config.theta,
            betas_logscale=smal_config.betas_logscale,
        )

        verts = verts @ align_rotation.T.unsqueeze(0)

        verts = verts * smal_config.scale + smal_config.trans

        source_meshes = Meshes(
            verts=verts,
            faces=torch.from_numpy(
                smal_config.smal_faces.astype(np.int64)
            ).to(device=device).long()[None]
        )

        source_verts = sample_points_from_meshes(source_meshes, 3000)

        target_mesh = Meshes(
            verts=torch.from_numpy(
                target_mesh_tuple[0]
            ).to(device).float()[None],
            faces=torch.from_numpy(
                target_mesh_tuple[1]
            ).to(device).long()[None],
        )

        target_verts = sample_points_from_meshes(target_mesh, 3000)

        if (visualize_interval >= 0) and (epoch % visualize_interval == 0):
            image = plot_compare_obj(
                target_mesh_tuple[0],
                target_mesh_tuple[1],
                verts[0].detach().data.cpu(),
                smal_config.smal_faces,
            )
            cv2.imwrite('tmp_stage.jpg', image)

        loss_chamfer, _ = chamfer_distance(source_verts, target_verts)

        optimizer.zero_grad()
        loss_chamfer.backward()
        optimizer.step()


def init_smal_parameters(device) -> Tuple[
    nn.Parameter, nn.Parameter, nn.Parameter, nn.Parameter
]:
    '''
    initialize betas_logscale, scale, theta, trans. only scale with one. others
    are with zero.

    inputs:
    -------
    device: torch.device
        the device to put the parameters on.

    outputs:
    --------
    betas_logscale: torch.nn.Parameter
        the log scale of betas. for some part deformation.
    scale: torch.nn.Parameter
        the global scale of the smal model.
    theta: torch.nn.Parameter
        the rotation of the smal model.
    trans: torch.nn.Parameter
        the global translation of the smal model.
    '''
    betas_logscale = torch.nn.Parameter(
        torch.zeros(1, 6).to(device),
        requires_grad=True,
    )
    scale = torch.nn.Parameter(
        torch.ones(1, 1, 1).to(device),
        requires_grad=True,
    )
    theta = torch.nn.Parameter(
        torch.zeros(1, 105).to(device),
        requires_grad=True,
    )
    trans = torch.nn.Parameter(
        torch.zeros(1, 3).to(device),
        requires_grad=True,
    )
    return betas_logscale, scale, theta, trans


def save_smal_no_tail_faces(beta, betas_logscale, theta, smal_model):
    '''
    save the faces of smal model with no tail

    inputs:
    -------
    '''
    verts, joints, Rs, v_shaped = smal_model(
        beta=beta, theta=theta, betas_logscale=betas_logscale,
    )
    vertices = verts[0].detach().cpu().numpy()
    faces = smal_model.faces
    image = plot_obj(vertices, faces)
    cv2.imwrite('tmp_smal.jpg', image)
    no_tail_vertices, no_tail_faces = load_obj('./no_tail.obj')

    image = plot_compare_obj(vertices, faces, no_tail_vertices, no_tail_faces)
    cv2.imwrite('tmp_compare.jpg', image)

    # ordered_no_tail_faces : (7590, 3)
    _, ordered_no_tail_faces = filter_tail(
        vertices, no_tail_vertices, faces
    )
    np.save('smal_no_tail_faces.npy', ordered_no_tail_faces)


def transform_and_save_model(
    meta_save_path: str,
    save_path: str,
    smal_config: SMALConfig,
    smal_model: SMALModel
) -> None:
    '''
    forward SMAL model and save the result to save_path
    '''
    align_rotation = torch.tensor(
        [
            [0,   1, 0],
            [-1,  0, 0],
            [0,   0, 1],
        ],
        dtype=torch.float32,
        device=smal_model.device,
    )
    verts, _, _, _ = smal_model(
        beta=smal_config.beta,
        theta=smal_config.theta,
        betas_logscale=smal_config.betas_logscale,
    )
    verts = verts @ align_rotation.T.unsqueeze(0)
    verts = verts * smal_config.scale
    save_obj(
        verts[0].detach().data.cpu(),
        smal_config.smal_faces,
        save_path,
    )

    with open(meta_save_path, 'w', encoding='utf-8') as f:
        json.dump(smal_config.to_json(), f, indent=4)


def find_template_smal_mesh(anime_name: str) -> None:
    '''
    find the template smal mesh.
    '''
    # prepare parameters
    img_keyword = file_mapping[anime_name]
    # get the save path from anime_name
    save_path = os.path.join(
        DT4D_SMAL_TEMPLATE_DIR, f'{anime_name}.obj',
    )
    meta_save_path = os.path.join(
        DT4D_SMAL_TEMPLATE_DIR, f'{anime_name}.json',
    )
    if os.path.exists(save_path) and os.path.exists(meta_save_path):
        return

    # load target mesh
    target_mesh_path = os.path.join(
        DT4D_OBJ_DIR, f'{anime_name}/00000.obj',
    )
    target_mesh = load_obj(target_mesh_path)

    list_beta = get_beta_from_json(ANIMAL3D_TRAIN_JSON_FILE, img_keyword)

    # initialize the smal model
    device = torch.device('cuda:0')
    beta, smal_model = init_smal_model(list_beta, device)
    betas_logscale, scale, theta, trans = init_smal_parameters(device)

    # prepare variables for optimize
    with_tail = get_with_tail(ANIMAL3D_TRAIN_JSON_FILE, img_keyword)
    if 'elkML' in anime_name:  # a special case
        with_tail = False
    if with_tail:
        smal_faces = smal_model.faces
    else:
        smal_faces = np.load('smal_no_tail_faces.npy')
    smal_config = SMALConfig(
        beta=beta,
        betas_logscale=betas_logscale,
        scale=scale,
        smal_faces=smal_faces,
        theta=theta,
        trans=trans,
    )

    # stage 1: optimize scale trans
    optimizer = torch.optim.Adam(
        [scale, trans], lr=0.02, betas=(0.5, 0.999)
    )
    optimizer.zero_grad()
    optimize_smal_model(
        num_epoch=51,
        optimizer=optimizer,
        smal_config=smal_config,
        smal_model=smal_model,
        target_mesh_tuple=target_mesh,
        visualize_interval=-1,
    )

    # stage 2: optimize all parameters
    optimizer = torch.optim.Adam(
        [beta, theta, scale, trans, betas_logscale],
        lr=0.005, betas=(0.5, 0.999),
    )
    optimizer.zero_grad()
    optimize_smal_model(
        num_epoch=401,
        optimizer=optimizer,
        smal_config=smal_config,
        smal_model=smal_model,
        target_mesh_tuple=target_mesh,
        visualize_interval=-1,
    )

    # save the optimized resting mesh
    transform_and_save_model(
        meta_save_path=meta_save_path,
        save_path=save_path,
        smal_config=smal_config,
        smal_model=smal_model,
    )


def main():
    '''
    the main function for fit a initial smal model onto target mesh.
    '''
    # get the list of target mesh
    for anime_name in tqdm(file_mapping.keys()):
        find_template_smal_mesh(anime_name)


if __name__ == '__main__':
    main()
