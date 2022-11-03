import os
import time
import copy
import json
import logging
import datetime
import numpy as np
from pprint import pformat
from argparse import ArgumentParser

import pkbar
import torch
import torch.nn.functional as Fx
import pytorch_warmup as warmup
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from retrain.dataset import SoundDataset
from retrain.model import Cnn14
from retrain.utils.ops import Adam, AdamP, CosineAnnealingLR, ReduceLROnPlateau
from retrain.metrics import accuracy, precision, recall, f1, cfm, classification_report, evaluate


class AdvancedBarking_eval():
    def __init__(self, val_label):

        self.val_label = val_label
        self.task = "AdvancedBarking"

        self.batch_size = 128
        self.lr = 0.0001
        self.num_class = 4
        self.category = ['barking', 'howling', 'crying', 'others']

        self.weight_loss = None
        self.optimizer = 'adamp'
        self.scheduler = 'cosine'

        self.time_drop_width = 64
        self.time_stripes_num = 2
        self.freq_drop_width = 8
        self.freq_stripes_num = 2
        self.model_arch = 'cnn14'

        self.sr = 8000
        self.nfft = 200
        self.hop = 80
        self.mel = 64
        self.inp = 500
        self.normalize_num = 32768.0
        self.Tmax = 500 

        self.freq_norm = False
        self.freq_norm_global= False
        self.freq_norm_channel = False

        self.preload = True
        self.sampler = False
        self.warmup = False
        self.spec_aug = True

def pred_multi_label(outputs):
    prob = np.squeeze(outputs.detach().cpu().numpy()).tolist()
    preds = torch.where(outputs > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
    return prob, preds

def main():
    parser = ArgumentParser()
    # data or model path setting
    parser.add_argument("--val_label", default = '/home/easonchen/sound_automation_retrain/Data/testing_set.csv', type=str, help="validation label csvfile. ex: AdvancedBarking_MC_20210805_01_val.csv")    
    parser.add_argument("--model_path", type=str)    
    args = parser.parse_args()

    params = AdvancedBarking_eval(args.val_label)
    ###################
    # model preparing #
    ###################
    model = prepare_model(params, args.model_path)

    ##################
    # data preparing #
    ##################
    print("Preparing validation data...")
    val_dataset = SoundDataset(params, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, pin_memory=True)

    dataset_size = len(val_dataset)
    print("val size: {}".format(len(val_dataset)))

    input_shape = (1, val_dataset.shape[1], val_dataset.shape[2])
    print("input_shape : {}".format(input_shape))
    summary(model, input_shape)

    ##################
    # model training #
    ##################
    # start to train the model
    evaluate_model(model, params, val_dataloader, dataset_size)

def evaluate_model(model, params, dataloader, dataset_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    model.eval()
    # prediction and groundtruth label
    y_true, y_pred = [], []
    
    # iterative training
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            labels = np.array([np.array(x) for x in labels]).T
            labels = torch.from_numpy(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs_dict = model(inputs)
            outputs = outputs_dict['clipwise_output']
            _, preds = pred_multi_label(outputs)
            y_true.extend(labels.data.cpu().detach().numpy())
            y_pred.extend(preds.cpu().detach().numpy())
    # finish an epoch
    time_elapsed = time.time() - start_time
    print()
    # compute classification results in an epoch
    # print(np.array(y_pred).dtype)
    epoch_AP, epoch_acc = evaluate(y_true, y_pred)
    epoch_mAP, epoch_acc = np.mean(epoch_AP[0:3]), np.mean(epoch_acc[0:3])
    ###############################
    # Evaluation: finish training #
    ###############################
    # finish training
    time_elapsed = time.time() - since
    cfmatrix = cfm(y_true, y_pred)
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('val Acc: {:4f}'.format(epoch_acc))
    print('val mAP: {:4f}'.format(epoch_mAP))
    print(cfmatrix)
    print(classification_report(y_true, y_pred))
    # with open(os.path.join(log_path, "classification_report.txt"), "w") as f:
    #     f.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+"\n")
    #     f.write('Best val Acc: {:4f}'.format(best_acc)+"\n")
    #     f.write('Best val mAP: {:4f}'.format(best_mAP)+"\n")
    #     f.write(str(cfmatrix)+"\n")
    #     f.write(classification_report(best_true, best_pred)+"\n")



def prepare_model(params, model_path):
    # Different model arch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("build model...")
    model = Cnn14(params)
    # Load pretrain weight
    print("load weights...")
    pretrained_dict = torch.load(model_path)
    model.load_state_dict(pretrained_dict)
    model = model.to(device)
    return model


if __name__ == '__main__':
    main()