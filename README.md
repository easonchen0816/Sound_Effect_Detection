# Sound Automation Retrain

This repo contains the retraining process of Furbo Dog Nanny sound classication, such as Advanced Barking, HomeEmergency, GlassBreaking, etc.

## Environments

The codebase is developed with Python 3.7.3. Install requirements as follows:

```bash
pip install -r requirements.txt
```

## Retrain Process

1. [Download Dataset](#download-dataset)
2. [Append Data](#append-data)
3. [Retrain](#retrain)
4. [Evaluate](#evaluate)
5. [Update Dataset](#update-dataset)

## Download Dataset

We put our dataset on [AWS S3 (MLOps environment) - ai-data-dvc bucket](https://s3.console.aws.amazon.com/s3/buckets/ai-data-dvc?region=ap-northeast-1&tab=objects), such as [furbo_advance_barking_data](https://s3.console.aws.amazon.com/s3/buckets/ai-data-dvc?region=ap-northeast-1&prefix=furbo_advance_barking_data/&showversions=false), [Mini_AdvancedBarking_data](https://s3.console.aws.amazon.com/s3/buckets/ai-data-dvc?region=ap-northeast-1&prefix=Mini_AdvancedBarking_data/&showversions=false). And they're version-controlled by [Data-Version-Control](https://dvc.org).

```bash
$ bash download_dataset.sh [Device:Furbo,Mini] \
                           [AI_Sound_Feature:AdvancedBarking,HomeEmergency,GlassBreaking,JP_HomeEmergency] \
                           [Target_Directory]
```

For example, if we want to download Furbo AdvancedBarking data in `/KIKI/hucheng`, we can execute the following command:

```bash
$ bash download_dataset.sh Furbo \
                           AdvancedBarking \
                           /KIKI/hucheng
```

## Append Data

Before running append_data.sh, we should prepare a folder with .wav audio files and an annotation csv file. The annotation csv file format is following:

| path                      | label | remark  |
|---------------------------|-------|---------|
| 1597048883767177_0.wav    | 0     | Barking |
| 1597050423359233_0.wav    | 1     | Howling |

Here we start to prepare new data to retrain model. We should prepare a folder with .wav audio files and an annotation csv file. The annotation csv file format is path,label,remark (ex: 1631919610_1631919612.wav,2,Crying or 1631995774_1631995775.wav,1,Howling,music). The append_data.sh will copy new audio files to train/val folder in dvc, and combine new annotation csv file and train/val csv annotation file in dvc. 

```bash
$ bash append_data.sh [Target_Folder_In_DVC] \
                      [New_Audio_Folder] \
                      [Annotation_CSV_In_DVC] \
                      [New_Annotation_CSV] \
                      [Combine_Annotation_CSV_Name]
```

For example, we want to append data from `/KIKI/henry/add_mini_advancedbarking_data/train/` to `/KIKI/hucheng/mini_advancedbarking_data/train/` and combine annotation csv file `/KIKI/henry/add_mini_advancedbarking_data/meta/add.csv` and `/KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv`, then named it as `/KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_02_train.csv`, we can execute the following command:

```bash
$ bash append_data.sh /KIKI/hucheng/mini_advancedbarking_data/train/ \
                      /KIKI/henry/add_mini_advancedbarking_data/train/ \
                      /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv \
                      /KIKI/henry/add_mini_advancedbarking_data/meta/add.csv \
                      /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_02_train.csv
```

## Retrain

After preparing annotation file and audio files, we would like to train model now. 

```bash
$ bash retrain.sh [Train Task:AdvancedBarking,GlassBreaking,HomeEmergency,HomeEmergency_JP,FCN,Integrate] \
                  [Training csv file] \
                  [Validation csv file] \
                  [Path to save training log and model wight] \
                  [Exp name prefix] \
                  [Training total epoch] \ 
                  [Pretrained weights path] \
                  [CUDA_DEVICE_ID]
```

For example, we want to retrain `AdvancedBarking` (Train Task) task use training data `AdvancedBarking_MC_20210805_01_train.csv` (Training csv file) and validation data `AdvancedBarking_MC_20210805_01_val.csv` (Validation csv file). And this experiment will save log and model weight in `/KIKI/hucheng/mini_advancedbarking_exp/` (Path to save training log and model wight). 
This retrain experiment named as `mini_advancedbarking_20210805_{}` (Exp name prefix). During training, we use pretraind model `/home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth` (Pretrained weights path) and train `200` (Training total epoch) epoch with `gup1` (CUDA_DEVICE_ID).

```bash
$ bash retrain.sh AdvancedBarking \
                  /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv \
                  /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_val.csv \
                  /KIKI/hucheng/mini_advancedbarking_exp/ \
                  mini_advancedbarking_20210805 \
                  200 \
                  /home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth
                  1
```

## Evaluate

After retraining the model, we would like to evalute the performance of model. It provides the precision/recall/f1-score on testing set and trigger rate on regression set. You can download the current online model on [AWS S3 - ai-sound-model-weight](https://s3.console.aws.amazon.com/s3/buckets/ai-sound-model-weight?region=us-east-1&prefix=AdvBarking/&showversions=false).

```bash
$ bash evaluate.sh [Device:Furbo,Mini,Furbo3] \
                   [AI_Sound_Feature:AdvancedBarking HomeEmergency,GlassBreaking,JP_HomeEmergency,FCN] \
                   [Model_Path] \
                   [Num_Class] \
                   [CUDA_DEVICE_ID]
```

For example, if we want to evaluate the performance of `AdvanceBarking_FB_20210714_01.pkl` on `Furbo AdvancedBarking` and use `gpu0` , we can execute the following command:

```bash
$ bash evaluate.sh Furbo \
                   AdvancedBarking \
                   ./AdvanceBarking_FB_20210714_01.pkl \
                   4 \
                   0
```

And it will save the precision/recall/trigger results as json file in `./result`.

## Update Dataset

After we add data into train/val/test/meta, we can commit the change by dvc and git.

```bash
$ bash update_dataset.sh [Device:Furbo,Mini] \
                         [AI_Sound_Feature:AdvancedBarking HomeEmergency,GlassBreaking,JP_HomeEmergency] \
                         [Target_Directory] \ 
                         [Commit Messages]
```
For example, if we want to commit the data change in `Furbo AdvancedBarking`, we can execute the following command:

```bash
$ bash update_dataset.sh Furbo \
                         AdvancedBarking \
                         /KIKI/hucheng \ 
                         "add fasle positive data"
```
