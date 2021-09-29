#!/bin/bash
# $1: device [Furbo, Mini]
# $2: sound feature [AdvancedBarking, HomeEmergency, GlassBreaking, JP_HomeEmergency]
# $3: target path
# $4: commit messages
# usage: bash update_dataset.sh Mini \
#                               AdvancedBarking \
#                               /KIKI/hucheng \ 
#                               "add fasle positive data"

cd $3/${1,,}_${2,,}_data
dvc add ./train ./val ./test ./meta
git add train.dvc val.dvc test.dvc meta.dvc
git commit -m "$4"
git push
dvc push
