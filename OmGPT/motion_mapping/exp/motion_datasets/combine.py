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
combine the motion dataset together

'''
import re
from time import time
from typing import List

import clip
import numpy as np
import spacy
import torch
from torch.utils import data

NLP = spacy.load("en_core_web_md")


class CombineDataset(data.Dataset):
    '''
    combine the motion dataset together
    '''
    clip_model = None

    def __init__(self, smpl_t2m_dataset, smal_t2m_dataset) -> None:
        super().__init__()

        self.smpl_t2m_dataset = smpl_t2m_dataset
        self.smal_t2m_dataset = smal_t2m_dataset

        # clip
        clip_version = 'ViT-B/32'
        self.clip_model = self._load_and_freeze_clip(clip_version)


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

    @staticmethod
    def _tokenize_text(raw_text):
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

        max_text_len = 20

        default_context_length = 77
        context_length = max_text_len + 2  
        assert context_length < default_context_length

        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        )

        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype
        )
        texts = torch.cat([texts, zero_pad], dim=1)

        return texts

        # return self.clip_model.encode_text(texts).float()

    def __len__(self):
        return max(
            len(self.smpl_t2m_dataset), len(self.smal_t2m_dataset)
        )

    def __getitem__(self, index):
        smal_index = index % len(self.smal_t2m_dataset)
        smpl_index = index % len(self.smpl_t2m_dataset)

        return [
            self.smpl_t2m_dataset[smpl_index],
            self.smal_t2m_dataset[smal_index],
        ]

    @classmethod
    def _initialize_clip(cls):
        if cls.clip_model is None:
            clip_version = 'ViT-B/32'
            clip_model, _ = clip.load(
                clip_version, device='cpu',
                jit=False
            )  

            clip.model.convert_weights(  # type: ignore
                clip_model
            )  

            # Freeze CLIP weights
            clip_model.eval()
            for parameter in clip_model.parameters():
                parameter.requires_grad = False

            cls.clip_model = clip_model

    @classmethod
    def collate(cls, batch):
        '''
        the collate function for the combine motion dataset
        '''

        start_time = time()

        # the input for real smpl is motion, caption, offset, mask
        smpl_batch = [b[0] for b in batch]
        smpl_meta = {
            'mean': [b[6] for b in smpl_batch],
            'std': [b[7] for b in smpl_batch],
            'name': [b[8] for b in smpl_batch],
        }

        real_smpl_input, real_smpl_token = cls._collate_smpl(smpl_batch)

        # the input for real smal is motion, caption, offset, mask
        smal_batch = [b[1] for b in batch]
        smal_meta = {
            'mean': [b[6] for b in smal_batch],
            'std': [b[7] for b in smal_batch],
            'name': [b[8] for b in smal_batch],
        }

        real_smal_input, _ = cls._collate_smal(smal_batch)

        start_swap_time = time()
        fake_smal_caption, fake_smpl_caption = _swap_subjects(
            real_smpl_input['smpl_text'],
            real_smpl_token,
            real_smal_input['smal_text'],
            real_smpl_input['smpl_subjects'],
        )

        # tokenize the text after swapping the subject
        # tokenize is for multiple gpu training
        real_smpl_input['smpl_text'] = cls._tokenize_text(
            real_smpl_input['smpl_text']
        )
        real_smal_input['smal_text'] = cls._tokenize_text(
            real_smal_input['smal_text']
        )

        # the input for fake smpl is text, offset, mask
        fake_smpl_input = {
            'smpl_text': cls._tokenize_text(fake_smpl_caption),
            'smpl_offsets': real_smpl_input['smpl_offsets'],
            'smpl_mask': real_smal_input['smal_mask'],
        }

        # the input for fake smal is text, offset, mask
        fake_smal_input = {
            'smal_text': cls._tokenize_text(fake_smal_caption),
            'smal_offsets': real_smal_input['smal_offsets'],
            'smal_mask': real_smpl_input['smpl_mask'],
        }


        return {
            'real_smpl_input': real_smpl_input,
            'real_smal_input': real_smal_input,
            'fake_smpl_input': fake_smpl_input,
            'fake_smal_input': fake_smal_input,
            'smal_meta': smal_meta,
            'smpl_meta': smpl_meta,
        }

    @classmethod
    def _collate_smpl(cls, smpl_batch):
        '''
        the collate function for the smpl motion dataset
        '''
        smpl_motions = _collate_index(smpl_batch, 2).float()

        smpl_offsets = _collate_index(smpl_batch, 5).float()

        smpl_means = _collate_index(smpl_batch, 6).float()

        smpl_stds = _collate_index(smpl_batch, 7).float()

        smpl_mask = _collate_length(
            smpl_batch, 3, smpl_motions.shape[1]
        ).float()

        smpl_text = [b[0] for b in smpl_batch]
        smpl_token = [b[4] for b in smpl_batch]
        smpl_subjects = [b[9] for b in smpl_batch]

        return {
            'smpl_motions': smpl_motions,
            'smpl_offsets': smpl_offsets,
            'smpl_mask': smpl_mask,
            'smpl_text': smpl_text,
            'smpl_token': smpl_token,
            'smpl_means': smpl_means,
            'smpl_stds': smpl_stds,
            'smpl_subjects': smpl_subjects,
        }, smpl_token

    @classmethod
    def _collate_smal(cls, smal_batch):
        '''
        the collate function for the smal motion dataset
        '''
        smal_motions = _collate_index(smal_batch, 2).float()

        smal_offsets = _collate_index(smal_batch, 5).float()

        smal_means = _collate_index(smal_batch, 6).float()

        smal_stds = _collate_index(smal_batch, 7).float()

        smal_mask = _collate_length(
            smal_batch, 3, smal_motions.shape[1]
        ).float()

        smal_text = [b[0] for b in smal_batch]
        smal_token = [b[4] for b in smal_batch]

        return {
            'smal_motions': smal_motions,
            'smal_offsets': smal_offsets,
            'smal_mask': smal_mask,
            'smal_text': smal_text,
            'smal_token': smal_token,
            'smal_means': smal_means,
            'smal_stds': smal_stds,
        }, smal_token


def _split_string(input_string):
    return re.split('(?<=[a-z])(?=[A-Z0-9])', input_string)


def find_subject(text):
    ''' solution from chatgpt about how to find subject.
    '''

    doc = NLP(text)
    subjects = [tok for tok in doc if "subj" in tok.dep_]
    return subjects


def _swap_subjects(
    smpl_captions: List[str],
    smpl_tokens: List[str],
    smal_captions: List[str],
    smpl_subjects: List[str],
):
    '''
    swap the subject in smpl and smal
    '''
   

    assert len(smpl_subjects) == len(smpl_captions)

    # find smal's subject
    smal_subjects = []
    for smal_caption in smal_captions:
        smal_subject = smal_caption.split(' ')[1].split('\'')[0]
        smal_subjects.append(smal_subject)
    assert len(smal_subjects) == len(smal_captions)

    swapped_smpl_captions = []
    for smpl_caption, smpl_subject, smal_subject in zip(
        smpl_captions, smpl_subjects, smal_subjects
    ):
        if smpl_subject == '':
            swapped_smpl_captions.append(smpl_caption)
            continue
        swapped_smpl_captions.append(
            smpl_caption.replace(smpl_subject, smal_subject)
        )

    swapped_smal_captions = []
    for smal_caption, smal_subject, smpl_subject in zip(
        smal_captions, smal_subjects, smpl_subjects
    ):
        if smal_subject == '':
            swapped_smal_captions.append(smal_caption)
            continue
        if smpl_subject == '':
            swapped_smal_captions.append(
                smal_caption.replace(smal_subject, 'person')
            )
            continue
        swapped_smal_captions.append(
            smal_caption.replace(smal_subject, smpl_subject)
        )

    return swapped_smpl_captions, swapped_smal_captions


def _collate_length(batch, index, max_len):
    '''
    the collate function for a given index
    '''
    lengths = torch.tensor([b[index] for b in batch])
    masks = lengths_to_mask(lengths, max_len).unsqueeze(-1).unsqueeze(-1)
    return masks


def _collate_index(batch, index):
    '''
    the collate function for a given index
    '''
    list_tensor = [torch.from_numpy(b[index]) for b in batch]
    stack_tensor = torch.stack(list_tensor, dim=0)
    return stack_tensor


def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(
        lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1)  # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    return motion, cond

def t2m_collate(batch):

    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1),
        'text': b[2],  
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)
