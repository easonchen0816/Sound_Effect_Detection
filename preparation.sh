#!/bin/bash
# $1: source_csv_path
# $2: target_csv_path
# $3: num_data
# $4: CUDA device ID
# usage: bash preparation.sh "/home/henry/tomofun_modules/all_device_data/meta_furbo3_data.csv"\
#                         '/home/easonchen/sound_automation_retrain/Data/training_set.csv' \
#                         20000 \
#                         0
CUDA_VISIBLE_DEVICES=$4 python preparation.py --sourcepath $1 \
                                            --targetpath $2 \
                                            --num_data $3 
