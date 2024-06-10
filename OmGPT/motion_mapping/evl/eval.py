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
import json
import os
import random
from glob import glob
from os.path import join as pjoin

import clip
import numpy as np
import torch
from motion_datasets import get_dataloader  # =================================
from motion_datasets.utils import _lengths_to_mask  # =========================
from motion_models.architecture import MotionMapping  # =======================
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.metrics import calculate_diversity  # ==============================
from utils.metrics import calculate_top_k  # ==================================
from utils.metrics import euclidean_distance_matrix  # ========================
from utils.options import parse_eval_args  # ==================================
from utils.utils import load_on_gpu  # ========================================
from utils.utils import unload_to_cpu  # ======================================
from ylib.print_options import options_str  # =================================
from ylib.torch import safe_l2_normalize  # ===================================

MAA_OFFSET = '../../_data/DeformingThings4D/animals_maa_offsets/'


def evaluate_matching_score(eval_wrapper, motion_loaders, file):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})
    print('========== Evaluating Matching Score ==========')
    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        score_list = []
        all_size = 0
        matching_score_sum = 0
        top_k_count = 0
        # print(motion_loader_name)
        with torch.no_grad():
            for idx, batch in enumerate(motion_loader):
                word_embeddings, pos_one_hots, _, sent_lens, motions, m_lens, _ = batch
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=m_lens
                )
                dist_mat = euclidean_distance_matrix(text_embeddings.cpu().numpy(),
                                                     motion_embeddings.cpu().numpy())
                matching_score_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=3)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]

                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

            all_motion_embeddings = np.concatenate(
                all_motion_embeddings, axis=0)
            matching_score = matching_score_sum / all_size
            R_precision = top_k_count / all_size
            match_score_dict[motion_loader_name] = matching_score
            R_precision_dict[motion_loader_name] = R_precision
            activation_dict[motion_loader_name] = all_motion_embeddings

        print(
            f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}')
        print(
            f'---> [{motion_loader_name}] Matching Score: {matching_score:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i+1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def _load_motion_mapping(motion_mapping, args):
    '''
    load the pretrained SMPL autoencoder
    '''
    ckpt_path = pjoin(
        args.load_path, args.load_ckpt, f'{args.load_step:09d}.ckpt'
    )
    checkpoint = torch.load(
        ckpt_path, map_location=torch.device('cpu')
    )
    motion_mapping.load_state_dict(checkpoint)


def _get_match_score(
    text_embeddings, motion_embeddings
):
    '''
    get the match score of the generated motions
    '''
    dist_mat = euclidean_distance_matrix(
        text_embeddings.detach().cpu().numpy(),
        motion_embeddings.detach().cpu().numpy()
    )
    matching_score = dist_mat.trace()
    return matching_score


def _get_top_k(
    text_embeddings, motion_embeddings
):
    '''
    get top k of the generated motions
    '''
    dist_mat = euclidean_distance_matrix(
        text_embeddings.detach().cpu().numpy(),
        motion_embeddings.detach().cpu().numpy()
    )
    argsmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argsmax, top_k=3)
    return top_k_mat.sum(axis=0)


def generate_motions(data_loader, network, args):
    '''
    generate motions from the data loader and network.
    mean while saving the motions

    inputs:
    -------

    return:
    -------
    '''
    all_size = 0
    matching_score_sum = 0
    top_k_count = 0
    for batch_data in data_loader:
        batch_gpu_data = load_on_gpu(batch_data)
        batch_output = network(batch_gpu_data)

        # get the clip text embeding
        text_embeddings = batch_output['f_smal_t_emd']
        # get the motion embeding
        motion_embeddings = batch_output['f_smal_t_pred']

        # size
        all_size += text_embeddings.shape[0]
        # matching score
        matching_score_sum += _get_match_score(
            text_embeddings, motion_embeddings
        )

        # top k
        from ylib.torch import safe_l2_normalize
        top_k_count += _get_top_k(
            safe_l2_normalize(text_embeddings),
            safe_l2_normalize(motion_embeddings)
        )

    matching_score = matching_score_sum / all_size
    R_precision = top_k_count / all_size
    # match_score_dict[motion_loader_name] = matching_score
    # R_precision_dict[motion_loader_name] = R_precision
    # activation_dict[motion_loader_name] = all_motion_embeddings

    print(f'---> [] Matching Score: {matching_score:.4f}')
    # print(f'---> [] Matching Score: {matching_score:.4f}', file=file, flush=True)

    line = f'---> [] R_precision: '
    for i in range(len(R_precision)):
        line += '(top %d): %.4f ' % (i+1, R_precision[i])
    print(line)
    # print(line, file=file, flush=True)
    # r precision

    # multi modal dist
    # diversity
    # multi modality


def _get_list_caption_path(caption_path):
    '''
    take the folder of ood caption as input.
    '''
    name_file_path = pjoin(caption_path, 'ood.json')
    if os.path.exists(name_file_path):
        with open(name_file_path, 'r') as openfile:
            list_caption_path = json.load(openfile)
    else:
        list_caption_path = glob(pjoin(caption_path, '*/meta.json'))
        list_caption_path.sort()
        with open(name_file_path, 'w') as openfile:
            json.dump(list_caption_path, openfile, indent=4)

    return list_caption_path


def _get_list_generated_motion_path(generated_motion_dir):
    '''
    get the list of generated motion path.
    '''
    name_file_path = generated_motion_dir + '.json'
    if os.path.exists(name_file_path):
        with open(name_file_path, 'r', encoding='utf-8') as openfile:
            list_path = json.load(openfile)
    else:
        # TODO: in order to evaluate motion generated from Mengyi I need to
        # change the motion.npy to 0.npy
        list_path = glob(pjoin(generated_motion_dir, '*/motion.npy'))
        list_path.sort()
        with open(name_file_path, 'w', encoding='utf-8') as openfile:
            json.dump(list_path, openfile, indent=4)
    return list_path


def _check_generated_motion_number(args):
    '''
    check the name of motions generated matches the number of captions.
    '''

    # get the list of caption file paths
    list_caption_path = _get_list_caption_path(args.caption_path)
    list_motion_id = [item.split('/')[-2] for item in list_caption_path]

    # load the list of generated motions
    list_generated_motion_path = _get_list_generated_motion_path(
        args.generated_motion_dir
    )
    if bool(args.lossy_match):
        # if lossy match, we only keep the generated motion that has the same
        new_list_caption_path = []
        new_list_generated_motion_path = []
        for generated_motion_path in list_generated_motion_path:
            generated_motion_name = generated_motion_path.split('/')[-2]
            # caption_name = caption_path.split('/')[-2]
            if generated_motion_name in list_motion_id:
                # new_list_caption_path.append(caption_path)
                # new
                new_list_generated_motion_path.append(generated_motion_path)
                new_list_caption_path.append(
                    list_caption_path[
                        list_motion_id.index(generated_motion_name)
                    ]
                )
        return new_list_caption_path, new_list_generated_motion_path
        # for caption_path, generated_motion_path in zip(
        #     list_caption_path, list_generated_motion_path
        # ):
        #     # caption_name = caption_path.split('/')[-2]
        #     generated_motion_name = generated_motion_path.split('/')[-2]
        #     if generated_motion_name in list_motion_id:
        #         new_list_caption_path.append(caption_path)
        #         new_list_generated_motion_path.append(generated_motion_path)
        #     if caption_name == generated_motion_name:
        #         new_list_caption_path.append(caption_path)
        #         new_list_generated_motion_path.append(generated_motion_path)
        # return new_list_caption_path, new_list_generated_motion_path
    else:
        for caption_path, generated_motion_path in zip(
            list_caption_path, list_generated_motion_path
        ):
            caption_name = caption_path.split('/')[-2]
            generated_motion_name = generated_motion_path.split('/')[-2]
            assert caption_name == generated_motion_name, (
                f'caption name {caption_name} does not match '
                f'generated motion name {generated_motion_name}'
            )

        print('check generated motion number done.')

        return list_caption_path, list_generated_motion_path


class EvaluationDataset(data.Dataset):
    '''
    evaluation dataset for motion generated and caption
    '''

    def _load_caption(self, caption_file_path):
        '''
        load caption from the caption file path.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        caption = meta['smal_text']
        return caption

    def _load_motion_length(self, caption_file_path):
        '''
        load motion length from the caption file path.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        motion_length = meta['motion_length']
        return motion_length

    def _load_gmotion(self, gmotion_file_path):
        '''
        load generated motion from the generated motion file path.
        '''
        motion = np.load(gmotion_file_path)
        return motion

    def _process_motion_to_maa(self, raw_motion):
        '''
        the current motion is represented as 321 and the motion should be 36*6
        '''
        if len(raw_motion.shape) == 3:
            if raw_motion.shape[1] < 196:
                motion = np.concatenate(
                    [
                        raw_motion,
                        np.zeros([196 - raw_motion.shape[0], 36, 6])
                    ],
                    axis=0
                )
            return motion
        assert len(raw_motion.shape) == 2, 'the motion should be 2d.'
        maa_len = 36 * 6
        motion = raw_motion[:, :maa_len]
        motion = motion.reshape(-1, 36, 6)

        if motion.shape[0] < 196:
            motion = np.concatenate(
                [motion, np.zeros([196 - motion.shape[0], 36, 6])], axis=0
            )
        return motion

    def _load_animal_id(self, caption_file_path):
        '''
        load animal id from meta file.
        '''
        with open(caption_file_path, 'r', encoding='utf-8') as openfile:
            meta = json.load(openfile)
        animal_id = meta['smal_id'].split('/')[-1].split('_')[0]
        return animal_id

    def _load_offset(self, animal_id):
        '''
        load offset from animal id
        '''
        offset_path = pjoin(MAA_OFFSET, f'{animal_id}', 'offset.pkl')
        offset = np.load(offset_path, allow_pickle=True)
        return offset

    def __init__(self, caption_path, gmotion_path) -> None:
        '''
        gmotion is short for generated motion
        '''
        self.data = []
        for caption_file_path, gmotion_file_path in tqdm(
            zip(caption_path, gmotion_path), desc='loading data',
        ):
            caption = self._load_caption(caption_file_path)
            gmotion = self._load_gmotion(gmotion_file_path)
            motion_length = self._load_motion_length(caption_file_path)
            maa_motion = self._process_motion_to_maa(gmotion)
            animal_id = self._load_animal_id(caption_file_path)
            offset = self._load_offset(animal_id)
            self.data.append(
                {
                    'caption': caption,
                    'motion': maa_motion,
                    'motion_length': motion_length,
                    'offset': offset,
                }
            )

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
        # Specific hardcoding for humanml dataset
        max_text_len = 20

        default_context_length = 77
        context_length = max_text_len + 2  # start_token + 20 + end_token
        assert context_length < default_context_length

        # [bs, context_length] # if n_tokens > context_length -> will truncate
        texts = clip.tokenize(
            raw_text, context_length=context_length, truncate=True
        )

        zero_pad = torch.zeros(
            [texts.shape[0], default_context_length - context_length],
            dtype=texts.dtype
        )
        texts = torch.cat([texts, zero_pad], dim=1)
        return texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
        get item from the dataset
        '''
        return self.data[index]

    @classmethod
    def collate(cls, batch):
        '''
        the collate function for the evaluation motion dataset
        '''
        list_text = [item['caption'][0] for item in batch]
        clip_token = cls._tokenize_text(list_text)
        list_motions = [torch.from_numpy(item['motion']) for item in batch]
        motions = torch.stack(list_motions, dim=0).float()

        # mask
        lengths = torch.tensor([item['motion_length'] for item in batch])
        mask = _lengths_to_mask(lengths, motions.shape[1])

        # offset
        list_offsets = [torch.from_numpy(item['offset']) for item in batch]
        offsets = torch.stack(list_offsets, dim=0)

        return {
            'clip_token': clip_token,
            'motions': motions,
            'offsets': offsets,
            'mask': mask,
        }


def _create_dataloader(sampled_caption_path, smapled_generated_motion_path):
    '''
    create dataloader from the caption path and generated motion path
    '''
    # create dataset
    dataset = EvaluationDataset(
        sampled_caption_path, smapled_generated_motion_path
    )
    # create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=dataset.collate
    )

    return loader


def _sample_two_list(list0, list1, number_of_items):
    '''
    sample two list with same length.
    '''
    assert len(list0) == len(list1), 'two list must have same length.'

    # Get a sample of indices
    indices = random.sample(range(len(list0)), number_of_items)

    # Use the sampled indices to get items from both lists
    sampled_items_1 = [list0[index] for index in indices]
    sampled_items_2 = [list1[index] for index in indices]

    return sampled_items_1, sampled_items_2


def _compute_r(list_pred):
    '''
    compute r precision
    '''
    matching_score_sum = 0
    top_k_count = 0
    all_size = 0
    for pred in list_pred:
        dist_mat = euclidean_distance_matrix(
            safe_l2_normalize(pred['text_emd']).detach().cpu().numpy(),
            safe_l2_normalize(pred['text_emd_pred']).detach().cpu().numpy(),
        )
        matching_score_sum += dist_mat.trace()

        argsmax = np.argsort(dist_mat, axis=1)
        top_k_mat = calculate_top_k(argsmax, top_k=3)
        top_k_count += top_k_mat.sum(axis=0)

        all_size += pred['text_emd'].shape[0]

    matching_score = matching_score_sum / all_size
    r_precision = top_k_count / all_size

    return matching_score, r_precision


def _compute_diversity(list_pred, diversity_times):
    '''
    compute diversity 
    '''
    # compute the diversity, diversity is defined as the average euclidean
    # distance between the generated motions.
    list_activation = []
    for pred in list_pred:
        list_activation.append(
            pred['motion_latent'].reshape(32, -1).detach().cpu().numpy()
        )
    activation = np.concatenate(list_activation, axis=0)
    if diversity_times > activation.shape[0]:
        diversity_times = activation.shape[0]
    diversity = calculate_diversity(activation, diversity_times)

    return diversity


def _eval_one_epoch(loader, model):
    '''
    evaluate the model for one epoch.
    '''
    list_pred = []
    with torch.no_grad():
        for data in loader:
            batch_gpu_data = load_on_gpu(data)
            pred = model.module.forward_for_evaluation(batch_gpu_data)
            list_pred.append(pred)

    # compute r precision for the generate motion
    matching_score, r_precision = _compute_r(list_pred)

    # compute diversity
    diversity = _compute_diversity(list_pred, diversity_times=30)

    return matching_score, r_precision, diversity


def main():
    '''
    main function
    '''
    # set args
    args, parser = parse_eval_args()
    print(options_str(args, parser))

    # check that the caption folder's name matches the generated motion
    # folder's names
    list_caption_path, list_generated_motion_path = _check_generated_motion_number(
        args
    )

    # load the trained model
    motion_mapping = MotionMapping(args)
    motion_mapping = nn.DataParallel(motion_mapping)
    _load_motion_mapping(motion_mapping, args)
    motion_mapping.to('cuda:0')
    motion_mapping.eval()

    # we do evaluation for 20 times
    num_eval = 1024
    list_ms = []
    list_rp = []
    list_dv = []
    for current_eval_i in range(20):
        sampled_caption_path, smapled_generated_motion_path = _sample_two_list(
            list_caption_path, list_generated_motion_path, num_eval
        )

        # create dataloader from the names
        loader = _create_dataloader(
            sampled_caption_path, smapled_generated_motion_path
        )

        ms, rp, dv = _eval_one_epoch(loader, motion_mapping)
        list_ms.append(ms)
        list_rp.append(rp)
        list_dv.append(dv)

    # compute the mean
    ms_mean = np.mean(list_ms)
    ms_std = np.std(list_ms)
    rp_mean = np.mean(list_rp, axis=0)
    rp_std = np.std(list_rp, axis=0)
    dv_mean = np.mean(list_dv)
    dv_std = np.std(list_dv)
    print(
        f'matching score (mm-dist) mean: {ms_mean:.4f} std: {ms_std:.4f}'
    )

    # Create a formatted string for each top-k result
    formatted_results = [
        f'{mean:.3f}\u00B1{std:.3f}' for mean, std in zip(rp_mean, rp_std)
    ]

    # Join the formatted results into a single string separated by spaces
    formatted_string = ' '.join(formatted_results)

    print(f'r precision mean: {formatted_string}')

    print(f'diversity mean: {dv_mean:.3f} std: {dv_std:.3f}')


if __name__ == '__main__':
    main()
