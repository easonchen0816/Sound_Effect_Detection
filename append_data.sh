#!/bin/bash
# $1: audio folder in dvc [ex: /KIKI/hucheng/mini_advancedbarking_data/train/, /KIKI/hucheng/mini_advancedbarking_data/val/]
# $2: new audio data folder [ex: /KIKI/henry/add_mini_advancedbarking_data/train/, /KIKI/henry/add_mini_advancedbarking_data/val/]
# $3: origin label csv files in dvc [ex: /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv]
# $4: new collected label csv file [ex: /KIKI/henry/add_mini_advancedbarking_data/meta/add.csv]
# $5: create a new label csv file in dvc meta [ex: /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_02_train.csv]


# usage: bash append_data.sh /KIKI/hucheng/mini_advancedbarking_data/train/ \
#                            /KIKI/henry/add_mini_advancedbarking_data/train/ \
#                            /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv \
#                            /KIKI/henry/add_mini_advancedbarking_data/meta/add.csv \
#                            /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_02_train.csv

# Remind
# $5 new label file name format: Task_Device_YYYYMMDD_Version_Type.csv
# Annotation in csv file format: row1 => [path, label, remark], row2 => [N_E8C74F19F988_1623297462.wav,0,barking]

python ./append_data/append_data.py --org_path ${1} --add_path ${2} --org_csv ${3} --add_csv ${4} --new_csv ${5}