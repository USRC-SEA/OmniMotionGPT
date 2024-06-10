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
from ylib.neural_networks.motion_auto_encoder import MotionEncoder  # =========
from ylib.neural_networks.transformer_backbone import TransformerEncoder  # ===
from ylib.skeleton_prior import smpl_key_joint_index  # =======================


class SemTransEncoder(nn.Module):
    '''
    the transformer encoder for semantic head
    '''

    def __init__(self, num_layers):
        super().__init__()

        self.encoder = TransformerEncoder(
            in_out_dim=(7 * 16, 512),
            d_model=512,
            num_layers=num_layers,
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
        )

    def forward(self, latent, mask):
        '''
        forward the latent
        '''
        feature = self.encoder(latent.view(-1, 49, 7 * 16))
        feature = feature * mask.squeeze(-1)
        max_feature = torch.max(feature, dim=1, keepdim=False)[0]
        clip_feature = self.mlp(max_feature)
        return clip_feature


class MotionCLIP(nn.Module):
    '''
    a network that maps motion to CLIP space
    '''

    def __init__(self, args) -> None:
        super().__init__()

        sem_head = args.sem_head

        # SMPL motion encoder
        self.smpl_motion_encoder = MotionEncoder(
            key_joint_index=tuple(smpl_key_joint_index),
            num_joints=22,
        )
        self.smpl_semantic_head = self._init_sem_head(sem_head)

        # clip
        clip_version = 'ViT-B/32'
        self.clip_model = self._load_and_freeze_clip(clip_version)

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

    def _encode_tokens(self, tokens):
        return self.clip_model.encode_text(tokens).float()

    def _init_sem_head(self, sem_head):
        '''
        initialize the semantic head
        '''
        if sem_head == 'mlp':
            print('===> using mlp semantic head')
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(49 * 7 * 16, 512),
                nn.ReLU(),
                nn.Linear(512, 512)  # 512 is hard coded for clip output size
            )

        if sem_head == 'linear':
            print('===> using linear semantic head')
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(49 * 7 * 16, 512),
            )

        if sem_head == 'mlp_1024_3':
            print('===> using mlp 1024 3 layers semantic head')
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(49 * 7 * 16, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
            )

        if sem_head == 'transen':
            print('===> using transformer encoder semantic head')
            return SemTransEncoder(self.num_sem_trasen_layers)

        raise NotImplementedError(f'Unknown semantic head {sem_head}')

    def forward(self, batch_data):
        '''
        forward function for the motion clip model

        inputs:
        -------
        batch_data: dict
            batch data
        '''
        # encode all the texts
        real_smpl_text_emd = self._encode_tokens(
            batch_data['real_smpl_input']['token_text'],
        )

        real_smpl_latent = self.smpl_motion_encoder(
            motions=batch_data['real_smpl_input']['smpl_motions'],
            offsets=batch_data['real_smpl_input']['smpl_offsets'],
        )

        real_smpl_text_emd_pred = self.smpl_semantic_head(
            real_smpl_latent
        )

        return {
            'real_smpl_text_emd': real_smpl_text_emd,
            'real_smpl_text_emd_pred': real_smpl_text_emd_pred,
        }


class LossCalculator(nn.Module):
    '''
    the loss calculator for the motion mapping model
    '''

    def __init__(self, args) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.l2_loss = lambda a, b: (a - b) ** 2
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def _compute_cos_loss(self, in_feat, out_feat):
        '''
        compute the cosine similarity loss
        '''
        features_norm = in_feat / in_feat.norm(dim=-1, keepdim=True)
        cos = self.cos_sim(features_norm, out_feat)
        cos_loss = (1 - cos).mean()
        return cos_loss

    def forward(self, batch_data, batch_output):
        '''
        forward function for the loss calculator
        '''
        # real semantic loss
        smpl_semantic_loss = self._compute_cos_loss(
            batch_output['real_smpl_text_emd'],
            batch_output['real_smpl_text_emd_pred']
        )

        total_loss = smpl_semantic_loss

        return total_loss, {
            'smpl_semantic_loss': smpl_semantic_loss,
        }
