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
        "--data_debug", type=int, default=0, help="debug flag in data loading"
    )
    group.add_argument("--batch_size", type=int,
                       default=128, help="batch size")


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
    group.add_argument(
        "--lm_type",
        type=str,
        default="mlp",
        help="type of linear mapping: linear, mlp",
    )
    group.add_argument(
        "--num_sem_trasen_layers",
        type=int,
        default=2,
        help="",
    )


def _add_optim(parser):
    """
    add options for optimization setting
    """
    group = parser.add_argument_group("optim")
    group.add_argument("--step", type=int, default=30001,
                       help="total training steps")


def _add_loss_weights(parser):
    """
    add the weights for each loss
    """
    group = parser.add_argument_group("loss_weights")
   

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


def _add_eval_path(parser):
    '''
    add evaluation path
    '''
    group = parser.add_argument_group("finetune_path")

    # load related
    group.add_argument(
        "--load_path", type=str,
        default="./", help="overall load path"
    )
    group.add_argument(
        "--load_ckpt", type=str, default="ckpt",
        help="checkpoint folder"
    )
    group.add_argument(
        "--load_step", type=int, help="load step"
    )

    # save related
    group.add_argument(
        "--save_path", type=str,
        default="./", help="overall save path"
    )
    group.add_argument(
        "--log", type=str, default="log",
        help="log folder"
    )
    group.add_argument(
        "--vis", type=str, default="vis",
        help="visualization folder"
    )
    group.add_argument(
        "--ckpt", type=str, default="ckpt",
        help="checkpoint folder"
    )

    # caption folder location
    group.add_argument(
        "--caption_path", type=str,
        default='/workspace_projects/intern/zhangsihao_yang/projects/animate_booth/_runtime/baselines/captions/ood',
        help="caption path"
    )

    # generated motion folder
    group.add_argument(
        "--generated_motion_dir", type=str, help="output path"
    )


def _check_and_make_eval_dirs(args):
    '''
    check the load file path exist.
    make the evaluation save path.
    '''
    ckpt_path = pjoin(
        args.load_path,
        args.load_ckpt,
        f"{args.load_step:09d}.ckpt"
    )
    assert os.path.exists(ckpt_path), f"ckpt file {ckpt_path} not found."

    _make_dirs(args)


def _add_eval_misc(parser):
    '''
    some evaluation about the lossy evalution
    '''
    group = parser.add_argument_group("misc")
    group.add_argument(
        "--lossy_match",
        type=int,
        default=0,
        help="whether the generated motion number has to match the caption number",
    )


def parse_eval_args():
    '''
    parse evaluation args
    '''
    parser = ArgumentParser()
    _add_eval_path(parser)
    _add_model(parser)
    _add_eval_misc(parser)

    args = parser.parse_args()

    # after parsing
    _check_and_make_eval_dirs(args)
    _set_seed()

    return args, parser
