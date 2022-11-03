#!/bin/bash
# $1: val .csv path
# $2: model path
# $3: CUDA device ID
# usage: ./evaluate.sh /home/easonchen/sound_automation_retrain/Data/testing_set.csv \ /RAID/eason/furbo_test/logs/snapshots/furbo_test_2022-09-14-17_57_lr-1e-04_optim-adamp_scheduler-cosine_spec_aug_/epoch_026_valloss_0.1220_valacc_0.9736_mAP_0.8489.pkl \ 1

CUDA_VISIBLE_DEVICES=$3 python evaluate.py --val_label $1 \
                                            --model_path $2 \