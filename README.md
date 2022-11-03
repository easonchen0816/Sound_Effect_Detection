# Sound Effect Detection

This repo contains the retraining process of Furbo Dog Nanny multi-label sound classication, such as Advanced Barking, HomeEmergency, GlassBreaking, etc.

## Environments

The codebase is developed with Python 3.7.3. Install requirements as follows:

```bash
pip install -r requirements.txt
```

## Retrain Process

1. [Preparation](#data-preparation)
2. [Retrain](#retrain)
3. [Evaluate](#evaluate)

## Data Preparation

```bash
$ bash preparation.sh source_csv_path \
                           target_csv_path \
                           num_data \
                           CUDA device ID
```

For example, if we want to create a SED annotation file with 20000 data from `/home/henry/tomofun_modules/all_device_data/meta_furbo3_data.csv` to `/home/easonchen/sound_automation_retrain/Data/training_set.csv` and use `gpu0`, we can execute the following command:

```bash
$ bash preparation.sh "/home/henry/tomofun_modules/all_device_data/meta_furbo3_data.csv"\
                        '/home/easonchen/sound_automation_retrain/Data/training_set.csv' \
                        20000 \
                        0
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




For example, we want to retrain `AdvancedBarking` (Train Task) task use training data `training_set.csv` (Training csv file) and validation data `testing_set.csv` (Validation csv file). And this experiment will save log and model weight in `/RAID/eason/furbo_test/logs/` (Path to save training log and model wight). 
This retrain experiment named as `furbo_test`. During training, we use pretraind model `/RAID/eason/pann_to_gtzan/Cnn14_mAP=0.431.pth` (Pretrained weights path) and train `200` (Training total epoch) epoch with `gup3` (CUDA_DEVICE_ID).

```bash
$ bash retrain.sh AdvancedBarking \ 
                    /home/easonchen/sound_automation_retrain/Data/training_set.csv \
                    /home/easonchen/sound_automation_retrain/Data/testing_set.csv \ 
                    /RAID/eason/furbo_test/logs/ \ 
                    furbo_test \ 
                    200 \ 
                    /RAID/eason/pann_to_gtzan/Cnn14_mAP=0.431.pth \ 
                    3
```

## Evaluate

After retraining the model, we would like to evalute the performance of model. It provides the precision/recall/f1-score on testing set and trigger rate on regression set. You can download the current online model on [AWS S3 - ai-sound-model-weight](https://s3.console.aws.amazon.com/s3/buckets/ai-sound-model-weight?region=us-east-1&prefix=AdvBarking/&showversions=false).

```bash
$ bash evaluate.sh [Val_Label] \
                   [Model_Path] \
                   [CUDA_DEVICE_ID]
```

For example, if we want to evaluate the performance of `AdvanceBarking_FB_20210714_01.pkl` on `Furbo AdvancedBarking` and use `gpu0` , we can execute the following command:

```bash
$ bash ./evaluate.sh /home/easonchen/sound_automation_retrain/Data/testing_set.csv \ 
        /RAID/eason/furbo_test/logs/snapshots/furbo_test_2022-09-14-17_57_lr-1e-04_optim-adamp_scheduler-cosine_spec_aug_/epoch_026_valloss_0.1220_valacc_0.9736_mAP_0.8489.pkl \ 
        0
```
