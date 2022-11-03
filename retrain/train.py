import os
import time
import copy
import json
import logging
import datetime
import numpy as np
from pprint import pformat
from argparse import ArgumentParser
import random
import pkbar
import torch
import torch.nn.functional as Fx
import pytorch_warmup as warmup
from torchsummary import summary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SoundDataset
from model import weak_mxh64_1024, Cnn14
from utils.ops import Adam, AdamP, CosineAnnealingLR, ReduceLROnPlateau
from metrics import accuracy, precision, recall, f1, cfm, classification_report, evaluate
from utils.losses import CrossEntropyLoss, BinaryCrossEntropyLoss, CrossEntropyLossWithoutWeight
from config import ParameterSetting_AdvancedBarking, ParameterSetting_GlassBreaking, ParameterSetting_HomeEmergency, ParameterSetting_HomeEmergency_JP, ParameterSetting_Integration

def fixed_seed(myseed):
    np.random.seed(myseed)
    random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)

def get_optim_scheduler(params, model, dataset_sizes):
    # optimizer
    if params.optimizer == "adam":
        optimizer = Adam(model.parameters(), lr=params.lr)
    elif params.optimizer == "adamp":
        optimizer = AdamP(model.parameters(), lr=params.lr)
    # scheduler
    if params.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, params.Tmax)
    elif params.scheduler == 'reduce':
        scheduler = ReduceLROnPlateau(optimizer)
    return optimizer, scheduler

def get_folder_name(params):
    # description of model and folder name
    now = datetime.datetime.now()
    folder_name = now.strftime("%Y-%m-%d-%H_%M")
    description = "_"
    if params.sampler:
        description += "sampler_"
    if params.warmup:
        description += "warmup_"
    if params.spec_aug:
        description += "spec_aug_"
    model_name = "{0:}_{1:}_lr-{2:.0e}_optim-{3:}_scheduler-{4:}{5:}".format(
                params.exp_name, folder_name, params.lr,
                params.optimizer, params.scheduler, description)
    save_model_path = os.path.join(params.save_root, "snapshots", model_name)
    return save_model_path, model_name

def pred_multi_label(outputs):
    prob = np.squeeze(outputs.detach().cpu().numpy()).tolist()
    preds = torch.where(outputs > 0.5, torch.tensor(1).cuda(), torch.tensor(0).cuda())
    return prob, preds

def train_model(model, params, dataloaders, dataset_sizes):
    ########################
    # Training opt setting #
    ########################
    fixed_seed(2700)
    optimizer, scheduler = get_optim_scheduler(params, model, dataset_sizes)
    if params.warmup:
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    ###########################################################
    # Use "exp_name" to create folder to save training result #
    ###########################################################
    save_model_path, model_name = get_folder_name(params)
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
        print("create folder: {}".format(save_model_path))

    log_path = os.path.join(params.save_root, "log", model_name)
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    ##########################
    # Save trainig parameter #
    ##########################
    argpath = os.path.join(log_path, 'parameter.txt')
    with open(argpath, 'w') as outfile:
        json.dump(vars(params), outfile)

    writer = SummaryWriter(log_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ######################
    # Validation metric  #
    ######################
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_mAP = 0.0
    best_true, best_pred = [], []


    ###################
    # Start training  #
    ###################
    for epoch in range(params.epochs):
        print('Epoch {}/{}'.format(epoch+1, params.epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # set model to train/eval model
            model.train() if phase == 'train' else model.eval()
            # set progress bar
            kbar = pkbar.Kbar(target=(dataset_sizes[phase]//params.batch_size)+1,
                              width=8)

            running_loss = 0.0
            running_corrects = 0
            # prediction and groundtruth label
            y_true, y_pred = [], []
            y_sin_true, y_sin_pred = [], []
            y_mul_true, y_mul_pred = [], []
            start_time = time.time()
            # iterative training
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                labels = np.array([np.array(x) for x in labels]).T
                # print(inputs.shape, labels.shape)
                labels = torch.from_numpy(labels)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs_dict = model(inputs)
                    outputs = outputs_dict['clipwise_output']
                    
                    _, preds = pred_multi_label(outputs)
                    # labels = labels.to(torch.float32)

                    # multi-labels loss
                    
                    loss = BinaryCrossEntropyLoss(outputs.to(torch.float32), labels.to(torch.float32))
                    
                    # backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        if params.warmup:
                            warmup_scheduler.dampen()

                # get output and pred in batch
                gt_label_in_batch = None
                gt_label_in_batch = labels.data.cpu().detach().numpy()
                running_loss += loss.item() * inputs.size(0)
                # print(np.array(y_true).shape)
                y_true.extend(gt_label_in_batch)
                y_sin_true.extend(gt_label_in_batch[np.sum(gt_label_in_batch, axis=1) == 1])
                y_mul_true.extend(gt_label_in_batch[np.sum(gt_label_in_batch, axis=1) != 1])
                # print(np.array(y_true).shape)
                preds = preds.cpu().detach().numpy()
                y_pred.extend(preds)
                y_sin_pred.extend(preds[np.sum(gt_label_in_batch, axis=1) == 1])
                y_mul_pred.extend(preds[np.sum(gt_label_in_batch, axis=1) != 1])

                if phase == 'train':
                    kbar.update(batch_idx, values=[("train loss in batch", loss)])
                    writer.add_scalar('train loss', loss, epoch*len(dataloaders[phase]) + batch_idx)
                else:
                    kbar.update(batch_idx, values=[("val loss in batch", loss)])
                    writer.add_scalar('val loss', loss, epoch*len(dataloaders[phase]) + batch_idx)

            # finish an epoch
            time_elapsed = time.time() - start_time
            print()
            print("finish this epoch in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            # compute classification results in an epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_AP, epoch_acc = evaluate(y_true, y_pred)
            epoch_mAP, epoch_acc = np.mean(epoch_AP[0:3]), np.mean(epoch_acc[0:3])
            
            if phase == 'train':
                kbar.add(1, values=[("train epoch loss", epoch_loss), ("train auc", epoch_acc), ("train mAP", epoch_mAP)])
                writer.add_scalar('train accuracy', epoch_acc, epoch)
                writer.add_scalar('train mAP', epoch_mAP, epoch)
            else:
                kbar.add(1, values=[("val epoch loss", epoch_loss), ("val auc", epoch_acc), ("val mAP", epoch_mAP)])
                writer.add_scalar('val accuracy', epoch_mAP, epoch)

                # save model if f1 and precision are all the best
                if epoch_mAP > best_mAP:
                    best_mAP = epoch_mAP 
                    best_acc = epoch_acc 
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_true = y_true
                    best_pred = y_pred
                    best_true_sin = y_sin_true
                    best_pred_sin = y_sin_pred
                    best_true_mul = y_mul_true
                    best_pred_mul = y_mul_pred
                    wpath = os.path.join(save_model_path, 'epoch_{:03d}_valloss_{:.4f}_valacc_{:.4f}_mAP_{:.4f}.pkl'.format(epoch+1, epoch_loss, epoch_acc, epoch_mAP))
                    torch.save(model.state_dict(), wpath)
                    print("=== save weight " + wpath + " ===")
                print()

    ###############################
    # Evaluation: finish training #
    ###############################
    # finish training
    time_elapsed = time.time() - since
    cfmatrix = cfm(best_true, best_pred)
    cfmatrix_sin = cfm(best_true_sin, best_pred_sin)
    cfmatrix_mul = cfm(best_true_mul, best_pred_mul)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val mAP: {:4f}'.format(best_mAP))
    print(cfmatrix)
    print(classification_report(best_true, best_pred))
    print(cfmatrix_sin)
    print(classification_report(best_true_sin, best_pred_sin))
    print(cfmatrix_mul)
    print(classification_report(best_true_mul, best_pred_mul))
    # load best model weights
    model.load_state_dict(best_model_wts)

    with open(os.path.join(log_path, "classification_report.txt"), "w") as f:
        f.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+"\n")
        f.write('Best val Acc: {:4f}'.format(best_acc)+"\n")
        f.write('Best val mAP: {:4f}'.format(best_mAP)+"\n")
        f.write(str(cfmatrix)+"\n")
        f.write(classification_report(best_true, best_pred)+"\n")


def prepare_model(params):
    # Different model arch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("build model...")
    if params.model_arch == 'vggish':
        model = weak_mxh64_1024(params.num_class,Fx.avg_pool2d,params)
    elif params.model_arch == "cnn14":
        model = Cnn14(params)
    else:
        raise(ValueError("Model arch must be vggish or cnn14."))

    # Load pretrain weight
    if params.pretrained:
        print("load pretrained weights...")
        if params.model_arch == "cnn14":
            pretrained_dict = torch.load(params.pretrained)['model']
        elif params.model_arch == 'vggish':
            pretrained_dict = torch.load(params.pretrained)
        else:
            raise(ValueError("Model arch must be vggish or cnn14."))

        # load partial pretrained weight
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    model = model.to(device)

    return model
