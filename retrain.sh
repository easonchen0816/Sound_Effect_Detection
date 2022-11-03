#!/bin/bash
# $1: training task [ex: AdvancedBarking, GlassBreaking, HomeEmergency, HomeEmergency_JP, Integrate]
# $2: training label csvfile. [ex: ex: AdvancedBarking_MC_20210805_01_train.csv]
# $3: validation label csvfile. [ex: ex: AdvancedBarking_MC_20210805_01_val.csv ]
# $4: the path to save training log and model wight [ex: /KIKI/hucheng/mini_advancedbarking_exp/]
# $5: Exp name prefix [ex: mini_advancedbarking_20210805]
# $6: training epoch number [ex: 200]
# $7: the path of pretrained weights [ex: /home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth]
# $8: gpu index [ex: 2]


# usage: bash retrain.sh AdvancedBarking \
#                        /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv \
#                        /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_val.csv \
#                        /KIKI/hucheng/mini_advancedbarking_exp/ \
#                        mini_advancedbarking_20210805 \
#                        200 \
#                        /home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth
#                        1

# Remind
# $1 training task must choose one of these name => AdvancedBarking, GlassBreaking, HomeEmergency, HomeEmergency_JP, FCN, Integrate
# other training parameter setting in ./retrain/config.py

CUDA_VISIBLE_DEVICES=${8} python ./retrain/retrain.py --task ${1} --train_label ${2} --val_label ${3} --save_root ${4} --exp_name ${5} --epochs ${6} --pretrained ${7}