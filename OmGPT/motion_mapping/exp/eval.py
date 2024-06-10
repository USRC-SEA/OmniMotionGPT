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
main function for get the clip features and visualize.

'''
import os
import pickle
import random
import time
from glob import glob
from os.path import join as pjoin

import numpy as np
import torch
import torch.optim as optim
from motion_datasets import get_smpl_dataloader  # ============================
from motion_datasets import get_smal_dataloader
from motion_models.motion_clip import LossCalculator  # =======================
from motion_models.motion_clip import MotionClip  # ===========================
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.loss_record import LossRecorder
from utils.options import parse_args
from ylib.deforming_things_4d.prior import smal_animal_names_v1_train  # ======
from ylib.maa import convert_maa_motion_to_positions
from ylib.print_options import options_str
from ylib.skeleton_obj_writer import draw_skeleton_animation  # ===============
from ylib.skeleton_obj_writer import draw_skeleton_animation_sbs_ps  # ========
from ylib.skeleton_obj_writer import draw_skeleton_animation_side_by_side  # ==
from ylib.skeleton_prior import smpl_22_parents  # ============================
from ylib.skeleton_prior import smal_35_kinematic_chain, smal_35_parents  # ===
from ylib.skeleton_prior import t2m_kinematic_chain as smpl_22_kinematic_chain

SMPL_MAA = '../../_data/MDM/maa/'
SMPL_MAA_META = '../../_data/MDM/maa_meta/'
SMAL_MAA = '../../_data/DeformingThings4D/animals_maa_motions/'


def _create_loss_recorder(args) -> LossRecorder:
    '''
    create the loss recorder
    '''
    log_path = pjoin(args.save_path, args.log)
    writer = SummaryWriter(log_path)
    loss_recoder = LossRecorder(writer)
    return loss_recoder


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


def log_losses(loss_recoder, loss, loss_details, step):
    '''
    log the losses to the SummaryWriter
    '''
    loss_recoder.add_scalar(name='loss', val=loss.item(), step=step)
    for key, val in loss_details.items():
        if isinstance(val, torch.Tensor):
            val = val.item()
        loss_recoder.add_scalar(
            name=f'{key}', val=val, step=step
        )


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


def _visualize_rec(
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


def visualize_results(
    batch_output, batch_gpu_data, save_path: str, vis_dir: str,
    step: int,
):
    '''
    visualize the results
    '''
    vis_path = pjoin(save_path, vis_dir, f'{step:09d}')
    os.makedirs(vis_path, exist_ok=True)

    batch_output_cpu = unload_to_cpu(batch_output)
    batch_input_cpu = unload_to_cpu(batch_gpu_data)

    # visualize smpl reconstruction
    _visualize_rec(
        input_batch=batch_input_cpu['real_smpl_input'],
        rec_motions=batch_output_cpu['smpl_rec'],
        out_offsets=batch_input_cpu['real_smpl_input']['smpl_offsets'],
        in_meta=batch_input_cpu['smpl_meta'],
        out_meta=batch_input_cpu['smpl_meta'],
        in_parents=smpl_22_parents,
        out_parents=smpl_22_parents,
        in_kinematic_chain=smpl_22_kinematic_chain,
        out_kinematic_chain=smpl_22_kinematic_chain,
        in_key='smpl_',
        save_dir=pjoin(vis_path, 'smpl_rec'),
    )

    


def train_one_epoch(
    data_loader, optimizer, neural_network, loss_calculator, step: int,
    loss_recoder, args
):
    '''
    train one epoch

    inputs:
    -------
    step
        the global current step
    '''
    finish_flag = False

    for batch_data in data_loader:
        batch_gpu_data = load_on_gpu(batch_data)
        optimizer.zero_grad()
        batch_output = neural_network(batch_gpu_data)
        loss, loss_details = loss_calculator(batch_gpu_data, batch_output)

        loss.backward()
        optimizer.step()

        print(loss.item())

        # log the loss
        if step % args.freq_log == 0:
            log_losses(loss_recoder, loss, loss_details, step)
            print(f'{step:06d}: {loss.item():.6f}')

        # save the ckpt
        if step % args.freq_save == 0 and step != 0:
            torch.save(
                neural_network.state_dict(),
                pjoin(args.save_path,  args.ckpt, f'{step:09d}.ckpt')
            )

        # save the visualization
        if (step % args.freq_vis == 0 and step != 0) or args.debug_vis:
            visualize_results(
                batch_output, batch_gpu_data,
                args.save_path, args.vis, step,
            )

        step += 1

        if step > args.step:
            finish_flag = True
            return finish_flag, step

    return finish_flag, step


def eval_one_epoch(
    data_loader,
    # optimizer,
    neural_network,
    # loss_calculator,
    # step: int,
    # loss_recoder, args
):
    '''
    evaluate the model for one full epoch.
    '''
    list_text = []
    text_clip_arr = []
    for batch_data in data_loader:
        batch_gpu_data = load_on_gpu(batch_data)
        # optimizer.zero_grad()
        batch_output = neural_network(
            batch_gpu_data,
            clip_only=True,
        )

        batch_output_cpu = unload_to_cpu(batch_output)

        list_text.extend(batch_gpu_data['real_smpl_input']['smal_text'])

        text_clip_arr.append(batch_output_cpu['clip_text_emb'])

    # stack all the text clip
    text_clip_arr = np.concatenate(text_clip_arr, axis=0)

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn import datasets
    from sklearn.manifold import TSNE

    # Load a sample dataset (e.g., the digits dataset)

    X = text_clip_arr

    # Apply t-SNE on the data
    tsne = TSNE(n_components=2, random_state=142)
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE output
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                          c='b', cmap='jet', edgecolor='w', s=60)
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="upper right", title="Classes")
    plt.gca().add_artist(legend1)
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of digits dataset')
    plt.savefig('tmp.png')

    # calculate the distance map and compute mean and std.

    def pairwise_distances(X):
        """Compute pairwise Euclidean distances."""
        sum_square = np.sum(np.square(X), axis=1)
        distance_map = np.add(
            np.add(-2 * np.dot(X, X.T), sum_square).T, sum_square)
        # Taking the positive part. Small numerical artifacts can create negative entries.
        return np.sqrt(np.maximum(distance_map, 0))

    # Compute the distance map
    distance_map = pairwise_distances(text_clip_arr)

    # Mask the diagonal and flatten the upper triangular part since the matrix is symmetric
    mask = np.triu(np.ones_like(distance_map), k=1)
    non_diagonal_values = distance_map[mask > 0]

    # Calculate mean and std for non-diagonal parts
    mean_distance = np.mean(non_diagonal_values)
    std_distance = np.std(non_diagonal_values)

    print(f"Mean Distance (non-diagonal): {mean_distance}")
    print(f"Standard Deviation of Distance (non-diagonal): {std_distance}")


    l2_norm = np.linalg.norm(text_clip_arr, axis=-1, keepdims=True)

    normalized_tensor = text_clip_arr / (l2_norm + 1e-7)

    # Compute the distance map
    distance_map = pairwise_distances(normalized_tensor)

    # Mask the diagonal and flatten the upper triangular part since the matrix is symmetric
    mask = np.triu(np.ones_like(distance_map), k=1)
    non_diagonal_values = distance_map[mask > 0]

    # Calculate mean and std for non-diagonal parts
    mean_distance = np.mean(non_diagonal_values)
    std_distance = np.std(non_diagonal_values)

    print(f"Mean Distance (non-diagonal): {mean_distance}")
    print(f"Standard Deviation of Distance (non-diagonal): {std_distance}")

    log_dir = 'logs/smal_embedding'

    # Initialize the SummaryWriter
    writer = SummaryWriter(log_dir)

    # Convert your PCA data to a PyTorch tensor
    embedding_tensor = torch.tensor(text_clip_arr)


    label_img = None
    writer.add_embedding(
        embedding_tensor,
        metadata=list_text,
        label_img=label_img,
        tag='text_clip_arr_embeddings',
    )

    # Close the writer
    writer.close()

   
def main():
    '''
    main function for training the mapping between SMPL and SMAL
    '''
    # set args
    args, parser = parse_args()
    print(options_str(args, parser))

    # load data
    dataset = 'smal'
    if dataset == 'smal':
        data_loader = get_smal_dataloader(args)
    elif dataset == 'smpl':
        data_loader = get_smpl_dataloader(args)

    # create loss recorder
    loss_recoder = _create_loss_recorder(args)

    # load the network onto multiple gpus
    motion_clip = MotionClip()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        motion_clip = nn.DataParallel(motion_clip)
    motion_clip.to('cuda:0')

    # load the loss calculator
    loss_calculator = LossCalculator(args)

    optimizer = optim.Adam(
        motion_clip.parameters(),
        lr=0.0001, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0, amsgrad=False
    )

    step = 0
    finish_flag = False

    loss_recoder.start_timer()
    for _ in range(9999999):

        eval_one_epoch(data_loader, motion_clip)
        break


    loss_recoder.end_timer()


if __name__ == '__main__':
    main()
