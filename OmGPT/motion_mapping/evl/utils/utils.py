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
from os.path import join as pjoin

import torch
from ylib.maa import convert_maa_motion_to_positions
from ylib.skeleton_obj_writer import draw_skeleton_animation_sbs_ps  # ========


def load_on_gpu(data):
    '''
    recursively load all tensors in a nested dictionary structure onto the gpu.
    '''
    if isinstance(data, dict):
        return {k: load_on_gpu(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.cuda('cuda:0')
    return data


def unload_to_cpu(data):
    '''
    recursively load all tensors in a nested dictionary structure onto the gpu.
    '''
    if isinstance(data, dict):
        return {k: unload_to_cpu(v) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.detach().data.cpu().numpy()
    return data


def _recover_text_from_tokens(tokens: str) -> str:
    '''
    recover the text from the tokens.
    this function is used for create caption of the visualzation.
    '''
    text = tokens.split('_')
    text = [token.split('/')[0] for token in text]
    # remove sos, eos, unk from the text
    text = [token for token in text if token not in ['sos', 'eos', 'unk']]
    text = ' '.join(text)
    return text


def _recover_position_from_motion(
    motion, mean, std, offset, kinematic_chain, mask
):
    '''
    recover the position from the motion. the motion is normalized.
    '''
    dn_in_motion = motion * std + mean
    positions = convert_maa_motion_to_positions(
        dn_in_motion,
        offsets=offset,
        kinematic_chain=kinematic_chain,
    )
    input_mask_bool_i = mask.squeeze(1).squeeze(1).astype(bool)
    val_positions = positions[input_mask_bool_i]
    return val_positions


def visualize_rec(
    input_batch,
    rec_motions,
    out_offsets,
    in_meta,
    out_meta,
    in_parents,
    out_parents,
    in_kinematic_chain,
    out_kinematic_chain,
    in_key: str,
    save_dir: str,
):
    '''
    visualize the reconstruction

    inputs:
    -------
    rec_motions : [batch_size, num_frames, num_joints, ]
        the reconstructed motions
    '''
    os.makedirs(save_dir, exist_ok=True)

    list_joint1 = []
    list_joint2 = []
    list_parent1 = []
    list_parent2 = []
    list_save_gif_path = []
    list_caption1 = []
    list_caption2 = []

    figure_index = 0
    for (
        input_motion_i,
        input_mean_i,
        input_std_i,
        input_mask_i,
        input_token_i,
        rec_motion_i,
        rec_mean_i,
        rec_std_i,
        in_offset_i,
        out_offset_i,
    ) in zip(
        input_batch[f'{in_key}motions'],
        in_meta['mean'],
        in_meta['std'],
        input_batch[f'{in_key}mask'],
        input_batch[f'{in_key}token'],
        rec_motions,
        out_meta['mean'],
        out_meta['std'],
        input_batch[f'{in_key}offsets'],
        out_offsets,
    ):
        
        recoverd_in_motion_i = _recover_position_from_motion(
            motion=input_motion_i,
            mean=input_mean_i,
            std=input_std_i,
            offset=in_offset_i,
            kinematic_chain=in_kinematic_chain,
            mask=input_mask_i,
        )

        text_i = _recover_text_from_tokens(input_token_i)

        recovered_out_motion_i = _recover_position_from_motion(
            motion=rec_motion_i,
            mean=rec_mean_i,
            std=rec_std_i,
            offset=out_offset_i,
            kinematic_chain=out_kinematic_chain,
            mask=input_mask_i,
        )

        list_joint1.append(recoverd_in_motion_i)
        list_joint2.append(recovered_out_motion_i)
        list_parent1.append(in_parents)
        list_parent2.append(out_parents)
        list_save_gif_path.append(pjoin(save_dir, f'{figure_index:04d}.mp4'))
        list_caption1.append(text_i)
        list_caption2.append('')

        figure_index += 1

    draw_skeleton_animation_sbs_ps(
        list_joint1=list_joint1,
        list_joint2=list_joint2,
        list_parent1=list_parent1,
        list_parent2=list_parent2,
        list_save_gif_path=list_save_gif_path,
        list_caption1=list_caption1,
        list_caption2=list_caption2,
    )
