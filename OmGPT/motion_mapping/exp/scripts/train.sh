# train again after the data processing
cd /OmGPT/motion_mapping/exp/

CUDA_VISIBLE_DEVICES=6                  \
python3 train.py                        \
        --data_debug            0       \
        --save_path     ../../_runtime/motion_mapping/exp05/train05  \
        --batch_size            128     \
        --step                  30001   \
        --freq_vis              5000    \
        --freq_save             5000    \
        --lambda_rc             100     \
        --lambda_rcxyz          100     \
        --lambda_vel            100     \
        --num_layers            8       \
        --normalize_text_clip   0       \
        --train_clip            1
