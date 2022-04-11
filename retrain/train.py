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
from torchsampler import ImbalancedDatasetSampler
from torch.utils.tensorboard import SummaryWriter

from dataset import SoundDataset
from model import weak_mxh64_1024, Cnn14
from utils.ops import Adam, AdamP, CosineAnnealingLR, ReduceLROnPlateau
from metrics import accuracy, precision, recall, f1, cfm, classification_report
from utils.losses import CrossEntropyLoss, BinaryCrossEntropyLoss, CrossEntropyLossWithoutWeight
from config import ParameterSetting_AdvancedBarking, ParameterSetting_GlassBreaking, ParameterSetting_HomeEmergency, ParameterSetting_HomeEmergency_JP, ParameterSetting_Integration


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

def pred_result(params, outputs):
    prob = np.squeeze(Fx.softmax(outputs).detach().cpu().numpy()).tolist()
    _, preds = torch.max(Fx.softmax(outputs), 1)
    return prob, preds

def train_model(model, params, dataloaders, dataset_sizes):
    ########################
    # Training opt setting #
    ########################
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
    best_precision = 0.0
    best_f1 = 0.0
    best_recall = 0.0
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
            start_time = time.time()
            # iterative training
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    prob, preds = pred_result(params, outputs)

                    # compute loss
                    if params.weight_loss is not None:
                        loss = CrossEntropyLoss(outputs, labels, params.weight_loss)
                    else:
                        loss = CrossEntropyLossWithoutWeight(outputs, labels)
                    
                    # backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        if params.warmup:
                            warmup_scheduler.dampen()

                # get output and pred in batch
                correct_in_batch = None
                gt_label_in_batch = None
                
                correct_in_batch = torch.sum(preds == labels.data)
                gt_label_in_batch = labels.data.cpu().detach().numpy()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += correct_in_batch
                acc_in_batch = correct_in_batch / float(len(inputs))

                y_true.extend(gt_label_in_batch)
                y_pred.extend(preds.cpu().detach().numpy())

                if phase == 'train':
                    kbar.update(batch_idx, values=[("train loss in batch", loss), ("train acc in batch", acc_in_batch)])
                    writer.add_scalar('train loss', loss, epoch*len(dataloaders[phase]) + batch_idx)
                else:
                    kbar.update(batch_idx, values=[("val loss in batch", loss), ("val acc in batch", acc_in_batch)])
                    writer.add_scalar('val loss', loss, epoch*len(dataloaders[phase]) + batch_idx)

            # finish an epoch
            time_elapsed = time.time() - start_time
            print()
            print("finish this epoch in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
            # compute classification results in an epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy(y_true, y_pred)
            epoch_f1 = f1(y_true, y_pred)
            epoch_recall = recall(y_true, y_pred)
            epoch_precision = precision(y_true, y_pred)
            
            if phase == 'train':
                kbar.add(1, values=[("train epoch loss", epoch_loss), ("train acc", epoch_acc), ("train precision", epoch_precision), ("train recall", epoch_recall), ("train f1", epoch_f1)])
                writer.add_scalar('train accuracy', epoch_acc, epoch)
                writer.add_scalar('train precision', epoch_precision, epoch)
                writer.add_scalar('train recall', epoch_recall, epoch)
                writer.add_scalar('train f1 score', epoch_f1, epoch)
            else:
                kbar.add(1, values=[("val epoch loss", epoch_loss), ("val acc", epoch_acc), ("val precision", epoch_precision), ("val recall", epoch_recall), ("val f1", epoch_f1)])
                writer.add_scalar('val accuracy', epoch_acc, epoch)
                writer.add_scalar('val precision', epoch_precision, epoch)
                writer.add_scalar('val recall', epoch_recall, epoch)
                writer.add_scalar('val f1 score', epoch_f1, epoch)

                # save model if f1 and precision are all the best
                if epoch_f1 > best_f1 or epoch_precision > best_precision or epoch_recall > best_recall or epoch_acc > best_acc or (epoch in [199, 174, 149, 124, 99, 74, 49, 24]):
                    best_f1 = epoch_f1 if epoch_f1 > best_f1 else best_f1
                    best_precision = epoch_precision if epoch_precision > best_precision else best_precision
                    best_acc = epoch_acc if epoch_acc > best_acc else best_acc
                    best_recall = epoch_recall if epoch_recall > best_recall else best_recall
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_true = y_true
                    best_pred = y_pred
                    wpath = os.path.join(save_model_path, 'epoch_{:03d}_valloss_{:.4f}_valacc_{:.4f}_pre_{:.4f}_recall{:.4f}_f1_{:.4f}.pkl'.format(epoch+1, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))
                    torch.save(model.state_dict(), wpath)
                    print("=== save weight " + wpath + " ===")
                print()

    ###############################
    # Evaluation: finish training #
    ###############################
    # finish training
    time_elapsed = time.time() - since
    cfmatrix = cfm(best_true, best_pred)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Precision: {:4f}'.format(best_precision))
    print('Best val Recall: {:4f}'.format(best_recall))
    print('Best val F1: {:4f}'.format(best_f1))
    print(cfmatrix)
    print(classification_report(best_true, best_pred, params.category))
    # load best model weights
    model.load_state_dict(best_model_wts)

    with open(os.path.join(log_path, "classification_report.txt"), "w") as f:
        f.write('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60)+"\n")
        f.write('Best val Acc: {:4f}'.format(best_acc)+"\n")
        f.write('Best val Precision: {:4f}'.format(best_precision)+"\n")
        f.write('Best val Recall: {:4f}'.format(best_recall)+"\n")
        f.write('Best val F1: {:4f}'.format(best_f1)+"\n")
        f.write(str(cfmatrix)+"\n")
        f.write(classification_report(best_true, best_pred, params.category)+"\n")


def callback_get_label(dataset, idx):
    y = dataset.Y[idx]
    return y

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
