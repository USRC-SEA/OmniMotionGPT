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
import random
from argparse import ArgumentParser
from os.path import join as pjoin

import numpy as np
import torch


def _make_dirs(args):
    """
    this function is coupled with _add_path function. if _add_path is not used,
    this function should not be used.

    make all the directories that could be used in training. this could help
    with prevent the error of directory not found.
    """
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(pjoin(args.save_path, args.log), exist_ok=True)
    os.makedirs(pjoin(args.save_path, args.vis), exist_ok=True)
    os.makedirs(pjoin(args.save_path, args.ckpt), exist_ok=True)


def _add_path(parser):
    """
    add the path for data loading, model training, result saving
    """
    group = parser.add_argument_group("path")
    group.add_argument("--save_path", type=str,
                       default="./", help="overall save path")
    group.add_argument("--log", type=str, default="log", help="log folder")
    group.add_argument("--vis", type=str, default="vis",
                       help="visualization folder")
    group.add_argument("--ckpt", type=str, default="ckpt",
                       help="checkpoint folder")


def _add_data(parser):
    """
    add options for data loading
    """
    group = parser.add_argument_group("data")
    group.add_argument(
        "--data_debug", type=int,
        default=0, help="debug flag in data loading"
    )
    group.add_argument(
        "--batch_size", type=int,
        default=128, help="batch size"
    )


def _add_model(parser):
    """
    add options for model structure
    """
    group = parser.add_argument_group("model")
    group.add_argument(
        "--sem_head",
        type=str,
        default="linear",
        help="type of semantic head: linear, mlp",
    )
    group.add_argument(
        "--mapping_type",
        type=str,
        default='id',
        help="type of mapping network: id, mlp, linear",
    )
    group.add_argument(
        "--fake_lossy_vel",
        type=int,
        default=0,
        help="whether to use fake lossy velocity",
    )


def _add_optim(parser):
    """
    add options for optimization setting
    include the total number of steps for training.
    """
    group = parser.add_argument_group("optim")
    group.add_argument(
        "--step", type=int, default=30001,
        help="total training steps"
    )


def _add_loss_weights(parser):
    """
    add the weights for each loss
    """
    group = parser.add_argument_group("loss_weights")
    # group.add_argument("--rec", type=float,
    #                    default=1.0, help="Weight for reconstruction loss")

    # direct forward related loss
    group.add_argument("--smpl_rec",
                       type=float, default=1.0, help="weight for real smpl reconstruction loss")
    group.add_argument("--smal_rec",
                       type=float, default=1.0, help="weight for real smal reconstruction loss")
    group.add_argument("--smpl_cap_rec",
                       type=float, default=1.0, help="weight for real smpl caption reconstruction loss")
    group.add_argument("--smal_cap_rec",
                       type=float, default=1.0, help="weight for real smal caption reconstruction loss")
    group.add_argument("--smpl_sem",
                       type=float, default=1.0, help="weight for smpl semantic loss")
    group.add_argument("--smal_sem",
                       type=float, default=1.0, help="weight for smal semantic loss")
    group.add_argument("--smpl_global_trans",
                       type=float, default=1.0, help="weight for smpl global translation loss")
    group.add_argument("--smal_global_trans",
                       type=float, default=1.0, help="weight for smal global translation loss")
    group.add_argument("--smpl_cap_global_trans",
                       type=float, default=1.0, help="weight for smpl caption global translation loss")
    group.add_argument("--smal_cap_global_trans",
                       type=float, default=1.0, help="weight for smal caption global translation loss")

    # cross forward related loss
    group.add_argument("--fake_smpl_consis",
                       type=float, default=1.0, help="weight for fake smpl consistency loss")
    group.add_argument("--fake_smal_consis",
                       type=float, default=1.0, help="weight for fake smal consistency loss")
    group.add_argument("--fake_smpl_sem",
                       type=float, default=1.0, help="weight for fake smpl semantic loss")
    group.add_argument("--fake_smal_sem",
                       type=float, default=1.0, help="weight for fake smal semantic loss")
    group.add_argument("--fake_smpl_global_rot",
                       type=float, default=0.0, help="weight for global rotation loss for fake smpl")
    group.add_argument("--fake_smal_global_rot",
                       type=float, default=1.0, help="weight for global rotation loss for fake smal")
    group.add_argument("--fake_smpl_ee",
                       type=float, default=0.0, help="Weight for fake smpl end effector loss")
    group.add_argument("--fake_smal_ee",
                       type=float, default=100.0, help="Weight for fake smal end effector loss")
    # type of fake motion global rotation loss
    group.add_argument("--fake_smpl_global_rot_type",
                       type=str, default="6d", help="type of fake smpl global rotation loss")
    group.add_argument("--fake_smal_global_rot_type",
                       type=str, default="6d", help="type of fake smal global rotation loss")

    
    group.add_argument(
        "--rec_vel",
        type=float,
        default=0.0,
        help="Weight for angular velocity reconstruction loss",
    )
    group.add_argument(
        "--rec_xyz_vel",
        type=float,
        default=0.0,
        help="Weight for velocity reconstruction loss",
    )


def _set_seed(seed=42):
    """
    set the random seed for torch and numpy
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you have multi-GPUs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _add_misc(parser):
    """
    add some freq related options.
    """
    group = parser.add_argument_group("misc")
    group.add_argument(
        "--freq_log",
        type=int,
        default=10,
        help="log loss on tensorboard every log_freq steps",
    )
    group.add_argument(
        "--freq_vis", type=int, default=5000, help="visualize every vis_freq steps"
    )
    group.add_argument(
        "--freq_save",
        type=int,
        default=5000,
        help="save checkpoint every save_freq steps",
    )
    group.add_argument(
        "--debug_vis", type=int, default=0, help="debug visualzation functions"
    )


def parse_args():
    """
    parse args
    """
    parser = ArgumentParser()
    _add_path(parser)
    _add_data(parser)
    _add_model(parser)
    _add_optim(parser)
    _add_loss_weights(parser)
    _add_misc(parser)
    args = parser.parse_args()

    # after parsing
    _make_dirs(args)
    _set_seed()

    return args, parser


def _add_finetune_path(parser):
    """
    add finetune
    """
    group = parser.add_argument_group("finetune_path")

    # load related
    group.add_argument("--load_path", type=str,
                       default="./", help="overall load path")
    group.add_argument(
        "--load_ckpt", type=str, default="ckpt", help="checkpoint folder"
    )
    group.add_argument("--load_step", type=int, help="load step")

    # save related
    group.add_argument("--save_path", type=str,
                       default="./", help="overall save path")
    group.add_argument("--log", type=str, default="log", help="log folder")
    group.add_argument("--vis", type=str, default="vis",
                       help="visualization folder")
    group.add_argument("--ckpt", type=str, default="ckpt",
                       help="checkpoint folder")


def _check_and_make_finetune_dirs(args):
    """
    check the load file path exist.
    make the finetune save path.
    """
    ckpt_path = pjoin(args.load_path, args.load_ckpt,
                      f"{args.load_step:09d}.ckpt")
    assert os.path.exists(ckpt_path), f"ckpt file {ckpt_path} not found."

    _make_dirs(args)


def parse_finetune_args():
    """
    parse the args for finetuning
    """
    parser = ArgumentParser()
    _add_finetune_path(parser)
    _add_data(parser)
    _add_model(parser)
    args = parser.parse_args()

    # after parsing
    _check_and_make_finetune_dirs(args)
    _set_seed()

    return args, parser


def _add_motion_clip_model(parser):
    '''
    add args for motion clip model
    '''
    group = parser.add_argument_group("model")
    group.add_argument(
        "--normalize_text_clip",
        type=int,
        default=0,
        help="should be bool, whether to normalize the inupt text clip",
    )
    group.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help=(
            "number of transformer layers for decoder, in the paper"
            "the number is set to 8."
        ),
    )
    group.add_argument(
        "--train_clip",
        type=int,
        default=0,
        help="should be bool, whether to train the text clip model",
    )


def _add_motion_clip_loss_weights(parser):
    """
    add the weights for each loss. this is copied from motion clip.
    include lambda_rc, lambda_rcxyz, lambda_vel, lambda_velxyz
    """
    group = parser.add_argument_group("loss_weights")
    group.add_argument(
        "--lambda_rc",
        default=1.0, type=float, help="weight of the reconstruction orientation loss"
    )
    group.add_argument(
        "--lambda_rcxyz",
        default=1.0, type=float, help="weight of the reconstruction xyz loss"
    )
    group.add_argument(
        "--lambda_vel",
        default=1.0, type=float, help="weight of the vel divergence loss"
    )
    group.add_argument(
        "--lambda_velxyz",
        default=1.0, type=float, help="weight of the vel divergence loss"
    )


def parse_motion_clip_args():
    '''
    parser the args for training motion clip.
    '''
    parser = ArgumentParser()
    _add_path(parser)
    _add_data(parser)
    _add_motion_clip_model(parser)
    _add_optim(parser)
    _add_motion_clip_loss_weights(parser)
    _add_misc(parser)
    args = parser.parse_args()

    # after parsing
    _make_dirs(args)
    _set_seed()

    return args, parser
