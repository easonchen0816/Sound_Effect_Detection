#!/bin/bash
# $1: device [Furbo, Mini]
# $2: sound feature [AdvancedBarking, HomeEmergency, GlassBreaking, JP_HomeEmergency]
# $3: model path
# $4: number of class, AdvancedBarking:4, HomeEmergency/GlassBreaking/JP_HomeEmergency:2
# $5: CUDA device ID
# usage: bash evaluate.sh Furbo \
#                         AdvancedBarking \
#                         ./AdvanceBarking_FB_20210714_01.pkl \
#                         4 \
#                         0

CUDA_VISIBLE_DEVICES=$5 python evaluation.py --device $1 \
                                            --feature $2 \
                                            --model_path $3 \
                                            --num_class $4
