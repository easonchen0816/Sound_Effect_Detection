#!/bin/bash
# $1: device [Furbo, Mini]
# $2: sound feature [AdvancedBarking, HomeEmergency, GlassBreaking, JP_HomeEmergency]
# $3: target path
# usage: bash download_dataset.sh Mini \
#                                 AdvancedBarking \
#                                 /KIKI/hucheng 

git clone git@bitbucket.org:tomofun/${1,,}_${2,,}_data.git $3/${1,,}_${2,,}_data
cd $3/${1,,}_${2,,}_data
dvc pull
