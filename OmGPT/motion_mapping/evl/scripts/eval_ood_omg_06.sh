#!/bin/bash

cd /OmGPT/motion_mapping/evl/

CUDA_VISIBLE_DEVICES=4                                  \
python3 eval.py                                         \
--load_path                                             \
../../_runtime/motion_mapping/exp04/train16             \
--load_step                                             \
30000                                                   \
--save_path                                             \
../tmp/2023_11_11                                       \
--mapping_type                                          \
id                                                      \
--generated_motion_dir                                  \
../../_runtime/motion_mapping_eval/train06/ood/regular  \
--lm_type                                               \
t2mgpt                                                  \
--lossy_match                                           \
1
