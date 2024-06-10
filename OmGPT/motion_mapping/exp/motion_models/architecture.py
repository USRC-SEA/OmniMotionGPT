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
the architecture of the motion model

'''
import clip
import torch
from torch import nn
from ylib.neural_networks.motion_auto_encoder import MotionAutoEncoder  # =====
from ylib.skeleton import Skeleton  # =========================================
from ylib.skeleton import compute_forward  # ==================================
from ylib.skeleton_prior import smal_35_face_joint_indx  # ====================
from ylib.skeleton_prior import smal_ee_idx  # ================================
from ylib.skeleton_prior import smal_key_joint_index  # =======================
from ylib.skeleton_prior import smpl_22_face_joint_indx  # ====================
from ylib.skeleton_prior import smpl_ee_idx  # ================================
from ylib.skeleton_prior import smpl_key_joint_index  # =======================
from ylib.skeleton_prior import smal_35_kinematic_chain as smal_kc  # =========
from ylib.skeleton_prior import t2m_kinematic_chain as smpl_kc  # =============


class MotionMapping(nn.Module):
    '''
    the motion mapping model
    '''

    def __init__(self, args) -> None:
        super().__init__()

        sem_head = args.sem_head
        mapping_type = args.mapping_type

        # smpl part
        self.smpl_ae = MotionAutoEncoder(
            key_joint_index=smpl_key_joint_index,
            num_joints=22,
        )

        self.smpl_semantic_head = self._init_sem_head(sem_head)

        # smal part
        self.smal_ae = MotionAutoEncoder(
            key_joint_index=smal_key_joint_index,
            num_joints=35,
        )
        self.smal_semantic_head = self._init_sem_head(sem_head)

        self.smpl_to_smal_mapping = self._init_mapping(mapping_type)
        self.smal_to_smpl_mapping = self._init_mapping(mapping_type)

        # clip
        clip_version = 'ViT-B/32'
        self.clip_model = self._load_and_freeze_clip(clip_version)

        self.smpl_clip_to_inter = self._init_clip_to_inter()
        self.smal_clip_to_inter = self._init_clip_to_inter()

    def _init_mapping(self, mapping_type):
        '''
        initialize the mapping part
        '''
        if mapping_type == 'id':
            print('ident')
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )

    def _init_clip_to_inter(self):
        '''
        initialize the clip to intermediate
        '''
        return nn.Sequential(

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 49 * 7 * 16)
        )

    def _init_sem_head(self, sem_head):
        '''
        initialize the semantic head
        '''
        if sem_head == 'mlp':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(49 * 7 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, 512)  # 512 is hard coded for clip output size
            )
        elif sem_head == 'linear':
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(49 * 7 * 16, 512),
            )
        else:
            raise NotImplementedError(f'Unknown semantic head {sem_head}')

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

    def _encode_text(self, raw_text, device):
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
        context_length = max_text_len + 2  
        assert context_length < default_context_length


        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        ).to(device)

        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype, device=texts.device
        )
        texts = torch.cat([texts, zero_pad], dim=1)

        return self.clip_model.encode_text(texts).float()

    def _encode_tokens(self, tokens):
        return self.clip_model.encode_text(tokens).float()

    def _get_ee_vel(
        self, motion, mean, std, offset, kinematic_chain, ee_index
    ):
        '''
        get the end effector velocity from a motion

        inputs:
        -------
        motion  :   [bs, num_frames, num_joints + 1, 6]
            the motion
        mean    :   [bs, num_joints + 1, 6]
            the mean of the motion
        std     :   [bs, num_joints + 1, 6]
            the std of the motion
        offset  :   [bs, num_joints, 3]
            the offset of the motion
        '''

        denorm_motion = motion * std.unsqueeze(1) + mean.unsqueeze(1)


        positions = Skeleton.forward_kinematics_cont6d(
            cont6d_params=denorm_motion,
            offsets=offset,
            kinematic_tree=kinematic_chain,
            world=True
        )

       
        velocity = positions[:, 1:, ee_index] - positions[:, :-1, ee_index]

        return velocity, positions

    def _get_vel_and_xyz(self, motion, mean, std, offset, kinematic_chain):
        '''
        '''

        denorm_motion = motion * std.unsqueeze(1) + mean.unsqueeze(1)


        positions = Skeleton.forward_kinematics_cont6d(
            cont6d_params=denorm_motion,
            offsets=offset,
            kinematic_tree=kinematic_chain,
            world=True
        )
        velocity = positions[:, 1:, :] - positions[:, :-1, :]
        return velocity, positions

    def forward(self, batch_data):
        '''
        forward function for the motion mapping model

        inputs:
        -------
        batch_data
            a dict of dict of tensor or words
        '''

        real_smpl_text_emd = self._encode_tokens(
            batch_data['real_smpl_input']['smpl_text'],
        )
        real_smal_text_emd = self._encode_tokens(
            batch_data['real_smal_input']['smal_text'],
        )

        real_smpl_latent, real_smpl_rec = self.smpl_ae(
            motions=batch_data['real_smpl_input']['smpl_motions'],
            offsets=batch_data['real_smpl_input']['smpl_offsets'],
        )
        real_smpl_text_emd_pred = self.smpl_semantic_head(real_smpl_latent)

        real_smal_latent, real_smal_rec = self.smal_ae(
            motions=batch_data['real_smal_input']['smal_motions'],
            offsets=batch_data['real_smal_input']['smal_offsets'],
        )
        real_smal_text_emd_pred = self.smal_semantic_head(real_smal_latent)

        # fake smpl forward
        fake_smal_latent = self.smpl_to_smal_mapping(real_smpl_latent)
        fake_smal_rec = self.smal_ae.dec_forward(
            fake_smal_latent, batch_data['fake_smal_input']['smal_offsets']
        )
        fake_smal_rec_latent, _ = self.smal_ae.enc_forward(
            fake_smal_rec, batch_data['fake_smal_input']['smal_offsets']
        )
        fake_smal_text_emd_pred = self.smal_semantic_head(fake_smal_latent)
        fake_smal_text_emd = self._encode_tokens(
            batch_data['fake_smal_input']['smal_text'],
        )

        # fake smal forward
        fake_smpl_latent = self.smal_to_smpl_mapping(real_smal_latent)
        fake_smpl_rec = self.smpl_ae.dec_forward(
            fake_smpl_latent, batch_data['fake_smpl_input']['smpl_offsets']
        )
        fake_smpl_rec_latent, _ = self.smpl_ae.enc_forward(
            fake_smpl_rec, batch_data['fake_smpl_input']['smpl_offsets']
        )
        fake_smpl_text_emd_pred = self.smpl_semantic_head(fake_smpl_latent)
        fake_smpl_text_emd = self._encode_tokens(
            batch_data['fake_smpl_input']['smpl_text'],
        )

        # fake smal relative ee velocity and real smpl ee velocity
        real_smpl_ee_vel, _ = self._get_ee_vel(
            motion=batch_data['real_smpl_input']['smpl_motions'],
            mean=batch_data['real_smpl_input']['smpl_means'],
            std=batch_data['real_smpl_input']['smpl_stds'],
            offset=batch_data['real_smpl_input']['smpl_offsets'],
            kinematic_chain=smpl_kc,
            ee_index=smpl_ee_idx,
        )
        fake_smal_ee_vel, fake_smal_xyz = self._get_ee_vel(
            motion=fake_smal_rec,
            mean=batch_data['real_smal_input']['smal_means'],
            std=batch_data['real_smal_input']['smal_stds'],
            offset=batch_data['real_smal_input']['smal_offsets'],
            kinematic_chain=smal_kc,
            ee_index=smal_ee_idx,
        )

        # get xyz velocity
        real_smpl_xyz_vel, real_smpl_xyz = self._get_vel_and_xyz(
            motion=batch_data['real_smpl_input']['smpl_motions'],
            mean=batch_data['real_smpl_input']['smpl_means'],
            std=batch_data['real_smpl_input']['smpl_stds'],
            offset=batch_data['real_smpl_input']['smpl_offsets'],
            kinematic_chain=smpl_kc,
        )
        
        cap_smpl_inter = self.smpl_clip_to_inter(
            real_smpl_text_emd
        ).view(-1, 49, 7, 16)
        caption_smpl_rec = self.smpl_ae.dec_forward(
            cap_smpl_inter, batch_data['fake_smpl_input']['smpl_offsets']
        )
        cap_smal_inter = self.smal_clip_to_inter(
            real_smal_text_emd
        ).view(-1, 49, 7, 16)
        caption_smal_rec = self.smal_ae.dec_forward(
            cap_smal_inter, batch_data['fake_smal_input']['smal_offsets']
        )

        return {
            'smpl_rec': real_smpl_rec,
            'smal_rec': real_smal_rec,
            'r_smpl_t_emd': real_smpl_text_emd,
            'r_smal_t_emd': real_smal_text_emd,
            'r_smpl_t_pred': real_smpl_text_emd_pred,
            'r_smal_t_pred': real_smal_text_emd_pred,
            # for fake smal
            'f_smal_latent': fake_smal_latent,
            'f_smal_rec_latent': fake_smal_rec_latent,
            'f_smal_t_emd': fake_smal_text_emd,
            'f_smal_t_pred': fake_smal_text_emd_pred,
            'f_smal_rec': fake_smal_rec,
            # for fake smpl
            'f_smpl_latent': fake_smpl_latent,
            'f_smpl_rec_latent': fake_smpl_rec_latent,
            'f_smpl_t_emd': fake_smpl_text_emd,
            'f_smpl_t_pred': fake_smpl_text_emd_pred,
            'f_smpl_rec': fake_smpl_rec,
            # velocity
            'real_smpl_ee_vel': real_smpl_ee_vel,
            'fake_smal_ee_vel': fake_smal_ee_vel,
            # # real velocity
            # 'real_smpl_xyz_vel': real_smpl_xyz_vel,
            # 'real_smal_xyz_vel': real_smal_xyz_vel,
            # # pred velocity
            # 'real_smpl_xyz_vel_pred': real_smpl_xyz_vel_pred,
            # 'real_smal_xyz_vel_pred': real_smal_xyz_vel_pred,
            # for forward direction
            'real_smpl_xyz': real_smpl_xyz,
            'fake_smal_xyz': fake_smal_xyz,
            # caption reconstruction
            'caption_smpl_rec': caption_smpl_rec,
            'caption_smal_rec': caption_smal_rec,
        }

    def forward_from_caption(self, batch_data):
        '''
        forward function for the caption as input. go through smpl decoder.

        inputs:
        -------
        batch_data
            a dict of dict of tensor or words
        '''
        # real_smpl_text_emd : [bs, 512]
        real_smpl_text_emd = self._encode_tokens(
            batch_data['real_smpl_input']['smpl_text'],
        )

        # 1. Get the weight and bias
        clip_weight = self.smpl_semantic_head[1].weight
        clip_bias = self.smpl_semantic_head[1].bias

        # 2. Compute the pseudo-inverse of the weight matrix
        w_pseudo_inv = torch.pinverse(clip_weight)

        # 3. Use the pseudo-inverse to compute the input
        x_retrieved = torch.mm(
            w_pseudo_inv, (real_smpl_text_emd - clip_bias).T
        ).T

        caption_smpl_latent = x_retrieved.reshape(-1, 49, 7, 16)

        caption_smpl_rec = self.smpl_ae.dec_forward(
            caption_smpl_latent, batch_data['fake_smpl_input']['smpl_offsets']
        )

        return {
            'caption_smpl_rec': caption_smpl_rec,
        }


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


class LossCalculator(nn.Module):
    '''
    the loss calculator for the motion mapping model
    '''

    def __init__(self, args) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l2_loss = lambda a, b: (a - b) ** 2
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

        # direct forward related loss
        self.lambda_smpl_rec = args.smpl_rec
        self.lambda_smal_rec = args.smal_rec
        self.lambda_smpl_cap_rec = args.smpl_cap_rec
        self.lambda_smal_cap_rec = args.smal_cap_rec
        self.lambda_smpl_sem = args.smpl_sem
        self.lambda_smal_sem = args.smal_sem
        self.lambda_smpl_global_trans = args.smpl_global_trans
        self.lambda_smal_global_trans = args.smal_global_trans
        self.lambda_smpl_cap_global_trans = args.smpl_cap_global_trans
        self.lambda_smal_cap_global_trans = args.smal_cap_global_trans

        # cross forward related loss
        self.lambda_fake_smpl_consis = args.fake_smpl_consis
        self.lambda_fake_smal_consis = args.fake_smal_consis
        self.lambda_fake_smpl_sem = args.fake_smpl_sem
        self.lambda_fake_smal_sem = args.fake_smal_sem
        self.lambda_fake_smpl_global_rot = args.fake_smpl_global_rot
        self.lambda_fake_smal_global_rot = args.fake_smal_global_rot
        self.lambda_fake_smpl_ee = args.fake_smpl_ee
        self.lambda_fake_smal_ee = args.fake_smal_ee
        self.fake_smpl_global_rot_type = args.fake_smpl_global_rot_type
        self.fake_smal_global_rot_type = args.fake_smal_global_rot_type

        # the velocity reconstruction loss
        self.lambda_rec_vel = args.rec_vel
        self.lambda_rec_xyz_vel = args.rec_xyz_vel

        # whether to use lossy ee vel for fake smal
        self.use_lossy_ee_vel = bool(args.fake_lossy_vel)

    def masked_l2(self, pred, target, mask):
        '''
        calculate the masked l2 loss
        assuming a.shape == b.shape == bs, J, Jdim, seqlen
        assuming mask.shape == bs, 1, 1, seqlen

        inputs:
        -------
        '''
        loss = self.l2_loss(pred, target)
        # gives \sigma_euclidean over unmasked elements
        loss = sum_flat(loss * mask.float())
        n_entries = pred.shape[2] * pred.shape[3]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val.mean()

    def _compute_trans_loss(self, out_motion, in_motion):
        '''
        compute the global translation loss

        inputs:
        -------
        out_motion : [bs, num_frames, num_joints + 1, 6]
            the output motion
        in_motion : [bs, num_frames, num_joints + 1, 6]
            the input motion
        '''
        return self.mse(out_motion[:, :, -1, :], in_motion[:, :, -1, :])

    def _compute_cos_loss(self, in_feat, out_feat):
        '''
        compute the cosine similarity loss
        '''
        features_norm = in_feat / in_feat.norm(dim=-1, keepdim=True)
        cos = self.cos_sim(features_norm, out_feat)
        cos_loss = (1 - cos).mean()
        return cos_loss

    def _compute_vel_loss(self, out_motion, in_motion):
        '''
        compute the velocity loss
        '''
        out_vel = out_motion[:, 1:, :, :] - out_motion[:, :-1, :, :]
        in_vel = in_motion[:, 1:, :, :] - in_motion[:, :-1, :, :]
        return self.mse(out_vel, in_vel)

    def _compute_lossy_vel_loss(self, fake_ee_vel, real_ee_vel):
        '''
        compute the lossy velocity loss
        '''
        scale = torch.norm(real_ee_vel, dim=-1, keepdim=True)
        # apply the scale to the mse loss element wise
        return self.mse(fake_ee_vel * scale, real_ee_vel * scale)

    def _compute_fake_smal_forward(
        self,
        f_smal_rec,
        smpl_motions,
    ):
        f_smal_forward = compute_forward(smal_35_face_joint_indx, f_smal_rec)
        smpl_forward = compute_forward(smpl_22_face_joint_indx, smpl_motions)
        return self.mse(f_smal_forward, smpl_forward)

    def forward(self, batch_data, batch_output):
        '''
        forward function for the loss calculator

        inputs:
        -------
        batch_data
            a dict of dict of tensor or words
        '''
        # real reconstruction loss
        smpl_rec_loss = self.masked_l2(
            batch_output['smpl_rec'],
            batch_data['real_smpl_input']['smpl_motions'],
            batch_data['real_smpl_input']['smpl_mask']
        )
        smal_rec_loss = self.masked_l2(
            batch_output['smal_rec'],
            batch_data['real_smal_input']['smal_motions'],
            batch_data['real_smal_input']['smal_mask']
        )

        # real global translation loss
        smpl_trans_loss = self._compute_trans_loss(
            batch_output['smpl_rec'],
            batch_data['real_smpl_input']['smpl_motions'],
        )
        smal_trans_loss = self._compute_trans_loss(
            batch_output['smal_rec'],
            batch_data['real_smal_input']['smal_motions'],
        )

        # real semantic loss
        smpl_semantic_loss = self._compute_cos_loss(
            batch_output['r_smpl_t_pred'],
            batch_output['r_smpl_t_emd']
        )
        smal_semantic_loss = self._compute_cos_loss(
            batch_output['r_smal_t_pred'],
            batch_output['r_smal_t_emd']
        )

        # fake latent consistency loss
        fake_smal_latent_consistency_loss = self.mse(
            batch_output['f_smal_latent'],
            batch_output['f_smal_rec_latent']
        )
        fake_smpl_latent_consistency_loss = self.mse(
            batch_output['f_smpl_latent'],
            batch_output['f_smpl_rec_latent']
        )

        # fake semantic loss
        fake_smal_semantic_loss = self._compute_cos_loss(
            batch_output['f_smal_t_pred'],
            batch_output['f_smal_t_emd']
        )
        fake_smpl_semantic_loss = self._compute_cos_loss(
            batch_output['f_smpl_t_pred'],
            batch_output['f_smpl_t_emd']
        )

        # fake smal ee loss
        if not self.use_lossy_ee_vel:
            fake_smal_ee_loss = self.mse(
                batch_output['fake_smal_ee_vel'],
                batch_output['real_smpl_ee_vel']
            )
        else:
            fake_smal_ee_loss = self._compute_lossy_vel_loss(
                batch_output['fake_smal_ee_vel'],
                batch_output['real_smpl_ee_vel']
            )

        # fake smal global rotation loss
        if self.fake_smal_global_rot_type == '6d':
            fake_smal_global_rot_loss = self.mse(
                batch_output['f_smal_rec'][:, :, 0, :],
                batch_data['real_smpl_input']['smpl_motions'][:, :, 0, :]
            )
        elif self.fake_smal_global_rot_type == 'forward':
            fake_smal_global_rot_loss = self._compute_fake_smal_forward(
                batch_output['fake_smal_xyz'].reshape(-1, 35, 3),
                batch_output['real_smpl_xyz'].reshape(-1, 22, 3),
            )
            fake_smal_global_rot_loss_1 = self.mse(
                batch_output['f_smal_rec'][:, :, 0, :],
                batch_data['real_smpl_input']['smpl_motions'][:, :, 0, :]
            )
            fake_smal_global_rot_loss = (
                fake_smal_global_rot_loss +
                fake_smal_global_rot_loss_1
            )
        else:
            raise NotImplementedError(
                f'Unknown fake smal global rot type {self.fake_smal_global_rot_type}'
            )

        # real angular velocity loss
        if self.lambda_rec_vel > 0:
            real_smpl_vel_loss = self._compute_vel_loss(
                batch_data['real_smpl_input']['smpl_motions'],
                batch_output['smpl_rec'],
            )
            real_smal_vel_loss = self._compute_vel_loss(
                batch_data['real_smal_input']['smal_motions'],
                batch_output['smal_rec'],
            )
        else:
            real_smpl_vel_loss = 0
            real_smal_vel_loss = 0

        # real xyz velocity loss
        if self.lambda_rec_xyz_vel > 0:
            real_smpl_xyz_vel_loss = self._compute_vel_loss(
                batch_output['real_smpl_xyz_vel'],
                batch_output['real_smpl_xyz_vel_pred'],
            )
            real_smal_xyz_vel_loss = self._compute_vel_loss(
                batch_output['real_smal_xyz_vel'],
                batch_output['real_smal_xyz_vel_pred'],
            )
        else:
            real_smpl_xyz_vel_loss = 0
            real_smal_xyz_vel_loss = 0

        # caption reconstruction loss
        caption_smpl_rec_loss = self.masked_l2(
            batch_output['caption_smpl_rec'],
            batch_data['real_smpl_input']['smpl_motions'],
            batch_data['real_smpl_input']['smpl_mask']
        )
        caption_smal_rec_loss = self.masked_l2(
            batch_output['caption_smal_rec'],
            batch_data['real_smal_input']['smal_motions'],
            batch_data['real_smal_input']['smal_mask']
        )

        # real global translation loss
        caption_smpl_trans_loss = self._compute_trans_loss(
            batch_output['caption_smpl_rec'],
            batch_data['real_smpl_input']['smpl_motions'],
        )
        caption_smal_trans_loss = self._compute_trans_loss(
            batch_output['caption_smal_rec'],
            batch_data['real_smal_input']['smal_motions'],
        )

        total_loss = (
            self.lambda_smpl_rec * smpl_rec_loss +
            self.lambda_smal_rec * smal_rec_loss +
            self.lambda_smpl_cap_rec * caption_smpl_rec_loss +
            self.lambda_smal_cap_rec * caption_smal_rec_loss +
            self.lambda_smpl_sem * smpl_semantic_loss +
            self.lambda_smal_sem * smal_semantic_loss +
            self.lambda_smpl_global_trans * smpl_trans_loss +
            self.lambda_smal_global_trans * smal_trans_loss +
            self.lambda_smpl_cap_global_trans * caption_smpl_trans_loss +
            self.lambda_smal_cap_global_trans * caption_smal_trans_loss +
            self.lambda_fake_smpl_consis * fake_smpl_latent_consistency_loss +
            self.lambda_fake_smal_consis * fake_smal_latent_consistency_loss +
            self.lambda_fake_smpl_sem * fake_smpl_semantic_loss +
            self.lambda_fake_smal_sem * fake_smal_semantic_loss +
            self.lambda_fake_smal_global_rot * fake_smal_global_rot_loss +
            self.lambda_fake_smal_ee * fake_smal_ee_loss +
            self.lambda_rec_vel + real_smpl_vel_loss +
            self.lambda_rec_vel + real_smal_vel_loss +
            self.lambda_rec_xyz_vel + real_smpl_xyz_vel_loss +
            self.lambda_rec_xyz_vel + real_smal_xyz_vel_loss
        )

        return total_loss, {
            'smpl_rec_loss': smpl_rec_loss,
            'smal_rec_loss': smal_rec_loss,
            'smpl_semantic_loss': smpl_semantic_loss,
            'smal_semantic_loss': smal_semantic_loss,
            'fake_smal_latent_consistency_loss': fake_smal_latent_consistency_loss,
            'fake_smpl_latent_consistency_loss': fake_smpl_latent_consistency_loss,
            'fake_smal_semantic_loss': fake_smal_semantic_loss,
            'fake_smpl_semantic_loss': fake_smpl_semantic_loss,
            'smpl_trans_loss': smpl_trans_loss,
            'smal_trans_loss': smal_trans_loss,
            'fake_smal_ee_loss': fake_smal_ee_loss,
            'fake_smal_global_rot_loss': fake_smal_global_rot_loss,
            'real_smpl_vel_loss': real_smpl_vel_loss,
            'real_smal_vel_loss': real_smal_vel_loss,
            'real_smpl_xyz_vel_loss': real_smpl_xyz_vel_loss,
            'real_smal_xyz_vel_loss': real_smal_xyz_vel_loss,
            'caption_smpl_rec_loss': caption_smpl_rec_loss,
            'caption_smal_rec_loss': caption_smal_rec_loss,
            'caption_smpl_trans_loss': caption_smpl_trans_loss,
            'caption_smal_trans_loss': caption_smal_trans_loss,
        }
