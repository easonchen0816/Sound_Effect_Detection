# Sound Automation Retrain #

This repo contains the retraining process of Furbo Dog Nanny sound classication, such as Advanced Barking, HomeEmergency, GlassBreaking, etc.

## Retrain Process ##

1. [Download Dataset](#download-dataset)
2. [Append Data](#append-data)
3. [Retrain](#retrain)
4. [Evaluate](#evaluate)
5. [Update Dataset](#update-dataset)

## Download Dataset ##

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

## Append Data ##

```bash
$ append_data.sh
```

## Retrain ##

```bash
$ retrain.sh
```

## Evaluate ##

```bash
$ evaluate.sh
```

## Update Dataset ##

```bash
$ update_dataset.sh
```