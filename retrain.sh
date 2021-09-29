#!/bin/bash
# $1: training task [ex: AdvancedBarking, GlassBreaking, HomeEmergency, HomeEmergency_JP, Integrate]
# $2: the dvc root [ex: /KIKI/hucheng/mini_advancedbarking_data/]
# $3: the folder path of train/val csv files [ex: /KIKI/hucheng/mini_advancedbarking_data/meta/]
# $4: name prefix of csvfile, ending up with train or val [ex: if csv file is AdvancedBarking_MC_20210805_01_train.csv, prefix is AdvancedBarking_MC_20210805_01]
# $5: the path to save training log and model wight [ex: /KIKI/hucheng/mini_advancedbarking_exp/]
# $6: Exp name prefix [ex: mini_advancedbarking_20210805]
# $7: training epoch number [ex: 200]
# $8: the path of pretrained weights [ex: /home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth]
# $9: gpu index [ex: 2]


# usage: bash retrain.sh AdvancedBarking \
#                        /KIKI/hucheng/mini_advancedbarking_data/ \
#                        /KIKI/hucheng/mini_advancedbarking_data/meta/ \
#                        AdvancedBarking_MC_20210805_01 \
#                        /KIKI/hucheng/mini_advancedbarking_exp/ \
#                        mini_advancedbarking_20210805 \
#                        200 \
#                        /home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth
#                        1

# Remind
# $1 training task must choose one of these name => AdvancedBarking, GlassBreaking, HomeEmergency, HomeEmergency_JP, Integrate
# other training parameter setting in ./retrain/config.py

CUDA_VISIBLE_DEVICES=${9} python ./retrain/retrain.py --task ${1} --dvc_root ${2} --csv_root ${3} --name_prefix ${4} --save_root ${5} --exp_name ${6} --epochs ${7} --pretrained ${8}