import os
import time
from tqdm import tqdm
from argparse import ArgumentParser
from pprint import pformat
import pandas as pd
import logging
import tables
from collections import Counter, defaultdict
from sklearn.metrics import confusion_matrix, classification_report
import ast
import json

import torch
from torch.utils.data import DataLoader, Dataset

from config import ParameterSetting
from models import Cnn14

logger = logging.getLogger(__file__)

POSSIBLE_TP = {'AdvancedBarking':['Barking','Howling','Crying','dog_growling'],
               'HomeEmergency':['HomeEmergency','JP_HomeEmergency','CO_Smoke_Alert','JP_Alert','pi_pi_sound'],
               'GlassBreaking':['GlassBreaking','Glass_Breaking','glass_drop','metal_collision','glass_collision','dishes_collision'],
               'JP_HomeEmergency':['JP_HomeEmergency','HomeEmergency','CO_Smoke_Alert','JP_Alert','pi_pi_sound']}

LABEL_TABLE = {'AdvancedBarking':{'Barking':['Barking'],'Howling':['Howling'],'Crying':['Crying']},
                'HomeEmergency':{'HomeEmergency':['CO_Smoke_Alert','JP_Alert','HomeEmergency']},
                'GlassBreaking':{'GlassBreaking':['GlassBreaking','Glass_Breaking','glass_drop']},
                'JP_HomeEmergency':{'JP_HomeEmergency':['CO_Smoke_Alert','JP_Alert','HomeEmergency']}}

LABEL_LIST = {'AdvancedBarking':['Barking','Howling','Crying'],
                'HomeEmergency':['HomeEmergency'],
                'GlassBreaking':['GlassBreaking'],
                'JP_HomeEmergency':['JP_HomeEmergency']}

LABEL_NUM = {'AdvancedBarking':{0:'Barking',1:'Howling',2:'Crying',3:'Other'},
                'HomeEmergency':{0:'HomeEmergency',1:'Other'},
                'GlassBreaking':{0:'GlassBreaking',1:'Other'},
                'JP_HomeEmergency':{0:'JP_HomeEmergency',1:'Other'}}

class SoundDataset_h5(Dataset):
    def __init__(self, h5_root, filename='furbo_testing_set.h5'):
        self.X = tables.open_file(os.path.join(h5_root, filename)).root.data

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class LabelConverter:
    def __init__(self, feature, csvfile):
        self.feature = feature
        self.label_table = LABEL_TABLE[self.feature]
        self.possible_tp_table = POSSIBLE_TP[self.feature]
        self.label_list = LABEL_LIST[self.feature]
        self.gt, self.possible_gt = self.read_csv(csvfile)
    
    def read_csv(self, csvfile):
        df = pd.read_csv(csvfile)
        gt, possible_gt = defaultdict(list), defaultdict(list)
        for lb_list in self.label_list:
            for i in range(len(df)):
                labels = ast.literal_eval(df.iloc[i].remark)
                if any(label in self.label_table[lb_list] for label in labels):
                    gt[lb_list].append(0)
                else:
                    gt[lb_list].append(1)
                if any(label in self.possible_tp_table for label in labels):
                    possible_gt[lb_list].append(0)
                else:
                    possible_gt[lb_list].append(1)
        return gt, possible_gt

def change_label_num(y_pred, lb_list):
    individual_pred = []
    for label in y_pred:
        if lb_list == 'Barking':
            if label != 0:
                individual_pred.append(1)
            else:
                individual_pred.append(0)
        elif lb_list == 'Howling':
            if label != 1:
                individual_pred.append(1)
            else:
                individual_pred.append(0)
        elif lb_list == 'Crying':
            if label != 2:
                individual_pred.append(1)
            else:
                individual_pred.append(0)
    return individual_pred

def save_report(report, lb_list, name, model_name):
    model_name = os.path.basename(model_name).split('.')[0]
    with open(os.path.join(args.result_dir,'{}_{}_{}_classification_report.json'.format(model_name, lb_list, name)), 'w') as files:
        json.dump(report, files, indent=4)

def show_pr(gt, pred, lb_list, model_name, name='model_performance'):
    print('='*30 + lb_list + " " + name +'='*30)
    print(confusion_matrix(gt[lb_list], pred))
    print(classification_report(gt[lb_list], pred, digits=4, target_names=[lb_list,'Other']))
    report = classification_report(gt[lb_list], pred, digits=4, target_names=[lb_list,'Other'], output_dict=True)
    save_report(report, lb_list, name, model_name)

def inference(dataloader, device, model):
    y_pred, prob = [], []
    with torch.no_grad():
        since = time.time()
        for batch_idx, spec in tqdm(enumerate(dataloader)):
            spec = torch.tensor(spec, dtype=torch.float32).to(device)
            outputs = model(spec)
            _, preds = torch.max(outputs, 1)

            pred_label = preds.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()

            y_pred.extend(pred_label)
            prob.extend(outputs)

        time_elapsed = time.time() - since
        print('test complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return y_pred, prob

def main(args):
    params = ParameterSetting(batch_size=args.batch_size, num_class=args.num_class)

    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    # model preparing 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cnn14(params)
    print(model)

    # data preparing 
    test_dataset = SoundDataset_h5(args.h5_root, '{}_testing_set.h5'.format(args.device.lower()))
    regression_dataset = SoundDataset_h5(args.h5_root, '{}_regression_set.h5'.format(args.device.lower()))

    test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)
    regression_dataloader = DataLoader(regression_dataset, batch_size=params.batch_size, shuffle=False)
    
    print("the number of testing set : {}".format(len(test_dataset)))
    print("the number of regression set : {}".format(len(regression_dataset)))
    print("="*60)

    # label preparing 
    lb_converter = LabelConverter(args.feature, os.path.join(args.csv_root, '{}_testing_set.csv'.format(args.device.lower())))
    gt = lb_converter.gt
    possible_gt = lb_converter.possible_gt

    for lb_list in LABEL_LIST[args.feature]:
        print("the number of {} gt labels : {}".format(lb_list, len(gt[lb_list])))
        print(Counter(gt[lb_list]))
        print("the number of {} possible gt labels : {}".format(lb_list, len(possible_gt[lb_list])))
        print(Counter(possible_gt[lb_list]))

    #   testing & regression set 
    for model_idx, model_name in enumerate(args.model_path):
        model.load_state_dict(torch.load(model_name))
        model.eval()
        model = model.to(device)

        # testing set
        y_pred, prob = inference(test_dataloader, device, model)
        
        for lb_list in LABEL_LIST[args.feature]:
            individual_pred = y_pred.copy()
            if args.feature == 'AdvancedBarking':
                individual_pred = change_label_num(y_pred, lb_list)
            
            show_pr(gt, individual_pred, lb_list, model_name)
            show_pr(possible_gt, individual_pred, lb_list, model_name, name='model_performance_including_possible_TP')
        
        # regression set
        y_pred, prob = inference(regression_dataloader, device, model)
        
        trigger = Counter(y_pred)
        model_name = os.path.basename(model_name).split('.')[0]
        for key, value in LABEL_NUM[args.feature].items():
            trigger[value] = trigger.pop(key)
        with open(os.path.join(args.result_dir, '{}_trigger_rate.json'.format(model_name)), 'w') as files:
            json.dump(trigger, files, indent=4)

if __name__ == '__main__':
    parser = ArgumentParser()
    # arguments for evaluation
    parser.add_argument("--h5_root", type=str, default='/KIKI/datasets/audio_evaluation')
    parser.add_argument("--csv_root", type=str, default='/KIKI/datasets/audio_evaluation')
    parser.add_argument("--result_dir", type=str, default='./result')
    parser.add_argument("--device", type=str, default='Furbo', choices=['Furbo','Mini'])
    parser.add_argument("--feature", type=str, default='AdvancedBarking', choices=['AdvancedBarking','HomeEmergency','GlassBreaking','JP_HomeEmergency'])
    parser.add_argument("--model_path", nargs="+", default=['glass_breaking.pkl'])
    parser.add_argument("--batch_size", type=int, default=128, help="the batch size")
    parser.add_argument("--num_class", type=int, default=4, help="number of classes")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Arguments: %s", pformat(args))

    main(args)
