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

import clip
import torch
from torch import nn
from ylib.neural_networks.transformer_backbone import PositionalEncoding
from ylib.skeleton import Skeleton  # =========================================
from ylib.skeleton_prior import t2m_kinematic_chain as smpl_kc  # =============
from ylib.torch import safe_l2_normalize  # ===================================


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class MotionClip(nn.Module):
    '''
    the motion clip model for the motion generate task.
    '''

    def __init__(
        self,
        # njoints,
        # nfeats,
        # num_frames,
        args,
        latent_dim=512, ff_size=1024,
        # num_layers=8,
        num_heads=4,
        dropout=0.1, activation="gelu",
        ablation=None, **kargs
    ):
        '''
        args provides:
            normalize_text_clip
            num_layers
        '''
        super().__init__()

        self.normalize_text_clip = bool(args.normalize_text_clip)
        self.num_layers = args.num_layers


        self.latent_dim = latent_dim

        self.ff_size = ff_size
        # self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation

        # self.input_feats = self.njoints*self.nfeats
        self.input_feats = 23 * 6

        self.sequence_pos_encoder = PositionalEncoding(
            self.latent_dim, self.dropout
        )

        seqTransDecoderLayer = nn.TransformerDecoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=activation,
            batch_first=True,
        )
        self.seqTransDecoder = nn.TransformerDecoder(
            seqTransDecoderLayer,
            num_layers=self.num_layers
        )

        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)

        # clip
        clip_version = 'ViT-B/32'
        if bool(args.train_clip):
            self.clip_model = self._load_and_unfreeze_clip(clip_version)
        else:
            self.clip_model = self._load_and_freeze_clip(clip_version)

    def _load_and_unfreeze_clip(self, clip_version):
        clip_model, _ = clip.load(
            clip_version, device='cpu',
            jit=False
        ) 


        clip_model.train()
        for parameter in clip_model.parameters():
            parameter.requires_grad = True


        return clip_model

    def _load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(
            clip_version, device='cpu',
            jit=False
        )  

        clip.model.convert_weights(  # type: ignore
            clip_model
        )  
        
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad = False

        return clip_model

    def _get_xyz(self, motion, mean, std, offset, kinematic_chain):
        '''
        get the joints xyz positions.
        '''
        # denorm the motion
        denorm_motion = motion * std.unsqueeze(1) + mean.unsqueeze(1)

        positions = Skeleton.forward_kinematics_cont6d(
            cont6d_params=denorm_motion,
            offsets=offset,
            kinematic_tree=kinematic_chain,
            world=True
        )

        return positions

    def forward(
        self, batch, use_text_emb=False, clip_only=False
    ):
        '''
        TODO clean this code 
        '''
        z = self.clip_model.encode_text(
            batch['real_smpl_input']['token_text']
        ).float()
        if clip_only:
            return {
                'clip_text_emb': z
            }

        # normalize text embedding
        if self.normalize_text_clip:
            z = safe_l2_normalize(z)

     
        bs, latent_dim = z.shape
        mask = batch['real_smpl_input']['smpl_mask'].squeeze(
            -1
        ).squeeze(-1).long().bool()
   

        z = z[:, None]  # sequence of size 1  #

        timequeries = torch.zeros(bs, 196, latent_dim, device=z.device)

        timequeries = self.sequence_pos_encoder(timequeries)

        output = self.seqTransDecoder(
            tgt=timequeries, memory=z,
            tgt_key_padding_mask=~mask
        )

        output = self.finallayer(output).reshape(bs, 196, 23, 6)

        # zero for padded area
        output[~mask] = 0


        smpl_xyz = self._get_xyz(
            motion=batch['real_smpl_input']['smpl_motions'],
            mean=batch['real_smpl_input']['smpl_means'],
            std=batch['real_smpl_input']['smpl_stds'],
            offset=batch['real_smpl_input']['smpl_offsets'],
            kinematic_chain=smpl_kc,
        )

        smpl_xyz_pred = self._get_xyz(
            motion=output,
            mean=batch['real_smpl_input']['smpl_means'],
            std=batch['real_smpl_input']['smpl_stds'],
            offset=batch['real_smpl_input']['smpl_offsets'],
            kinematic_chain=smpl_kc,
        )

        return {
            'smpl_rec': output,
            'smpl_xyz': smpl_xyz,
            'smpl_xyz_pred': smpl_xyz_pred,
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

        self.lambda_rc = args.lambda_rc
        self.lambda_rcxyz = args.lambda_rcxyz
        self.lambda_vel = args.lambda_vel
        self.lambda_velxyz = args.lambda_velxyz

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

    def _compute_vel_loss(self, out_motion, in_motion):
        '''
        compute the velocity loss
        '''
        out_vel = out_motion[:, 1:, :, :] - out_motion[:, :-1, :, :]
        in_vel = in_motion[:, 1:, :, :] - in_motion[:, :-1, :, :]
        return self.mse(out_vel, in_vel)

    def forward(self, batch_data, batch_output):
        '''
        forward function for the loss calculator

        inputs:
        -------
        batch_data
            a dict of dict of tensor or words
        '''
        # real reconstruction loss
        rc_loss = self.masked_l2(
            batch_output['smpl_rec'],
            batch_data['real_smpl_input']['smpl_motions'],
            batch_data['real_smpl_input']['smpl_mask']
        )

        # reconstruction xyz loss
        rcxyz_loss = self.masked_l2(
            batch_output['smpl_xyz_pred'],
            batch_output['smpl_xyz'],
            batch_data['real_smpl_input']['smpl_mask'],
        )

        # reconstruction orientation velocity loss
        vel_loss = self._compute_vel_loss(
            batch_output['smpl_rec'],
            batch_data['real_smpl_input']['smpl_motions'],
        )

        total_loss = (
            self.lambda_rc * rc_loss +
            self.lambda_rcxyz * rcxyz_loss +
            self.lambda_vel * vel_loss
        )

        return total_loss, {
            'smpl_rec_loss': rc_loss,
            'smpl_xyz_loss': rcxyz_loss,
            'smpl_vel_loss': vel_loss,
        }
