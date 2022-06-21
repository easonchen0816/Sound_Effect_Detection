import os
import logging
from pprint import pformat
from argparse import ArgumentParser

from torchsummary import summary
from torch.utils.data import DataLoader
# from torchsampler import ImbalancedDatasetSampler
from imbalanced import ImbalancedDatasetSampler

from dataset import SoundDataset
from train import train_model, callback_get_label, prepare_model
from config import ParameterSetting_AdvancedBarking, ParameterSetting_GlassBreaking, ParameterSetting_HomeEmergency, ParameterSetting_HomeEmergency_JP, ParameterSetting_FCN, ParameterSetting_Integration


def main():
    parser = ArgumentParser()
    # data or model path setting
    parser.add_argument("--task",        type=str, help="training task. ex: AdvancedBarking, GlassBreaking, HomeEmergency, HomeEmergency_JP, Integrate")
    parser.add_argument("--train_label", type=str, help="training label csvfile. ex: AdvancedBarking_MC_20210805_01_train.csv")
    parser.add_argument("--val_label",   type=str, help="validation label csvfile. ex: AdvancedBarking_MC_20210805_01_val.csv")    
    parser.add_argument("--save_root",   type=str, help="the path to save training log and model wight. ex:/KIKI/hucheng/mini_advancedbarking_exp/")
    parser.add_argument("--exp_name",    type=str, help="Exp name prefix. ex: mini_advancedbarking_20210805")
    parser.add_argument("--epochs",      type=int, help="epoch number. ex: 200")
    parser.add_argument("--pretrained",  type=str, help="the path of pretrained weights. ex: /home/henry/pretrain_weight/Cnn14_8k_mAP=0.416.pth") 
    args = parser.parse_args()
    logger = logging.getLogger(__file__)
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    ##################
    # config setting #
    ##################
    if args.task == "AdvancedBarking":
        params = ParameterSetting_AdvancedBarking(args.train_label, args.val_label, args.save_root, args.exp_name, args.pretrained, args.epochs)
    elif args.task == "GlassBreaking":
        params = ParameterSetting_GlassBreaking(args.train_label, args.val_label, args.save_root, args.exp_name, args.pretrained, args.epochs)
    elif args.task == "HomeEmergency":
        params = ParameterSetting_HomeEmergency(args.train_label, args.val_label, args.save_root, args.exp_name, args.pretrained, args.epochs)
    elif args.task == "HomeEmergency_JP":
        params = ParameterSetting_HomeEmergency_JP(args.train_label, args.val_label, args.save_root, args.exp_name, args.pretrained, args.epochs)
    elif args.task == "FCN":
        params = ParameterSetting_FCN(args.train_label, args.val_label, args.save_root, args.exp_name, args.pretrained, args.epochs)
    elif args.task == "Integrate":
        params = ParameterSetting_Integration(args.train_label, args.val_label, args.save_root, args.exp_name, args.pretrained, args.epochs)
    else:
        raise(ValueError("Training task must be AdvancedBarking, GlassBreaking, HomeEmergency, HomeEmergency_JP or Integrate."))

    #######################################
    # Create foldr to save trainig result #
    #######################################
    if not os.path.exists(params.save_root):
        os.mkdir(params.save_root)
        print("create folder: {}".format(params.save_root))
    if not os.path.exists(os.path.join(params.save_root, 'snapshots')):
        os.mkdir(os.path.join(params.save_root, 'snapshots'))
    if not os.path.exists(os.path.join(params.save_root, 'log')):
        os.mkdir(os.path.join(params.save_root, 'log'))

    ###################
    # model preparing #
    ###################
    model = prepare_model(params)

    ##################
    # data preparing #
    ##################
    print("Preparing training data...")
    train_dataset = SoundDataset(params, train=True)
    print("Preparing validation data...")
    val_dataset = SoundDataset(params, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, pin_memory=True)
    if params.sampler:
        train_dataloader = DataLoader(train_dataset, sampler=ImbalancedDatasetSampler(train_dataset, callback_get_label=callback_get_label),
                                      batch_size=params.batch_size, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print("train size: {}, val size: {}".format(len(train_dataset), len(val_dataset)))

    input_shape = (1, train_dataset.shape[1], train_dataset.shape[2])
    print("input_shape : {}".format(input_shape))
    summary(model, input_shape)

    ##################
    # model training #
    ##################
    # start to train the model
    train_model(model, params, dataloaders, dataset_sizes)


if __name__ == '__main__':
    main()
