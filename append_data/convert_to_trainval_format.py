import os
import csv
import glob

import pandas as pd
from argparse import ArgumentParser


LABEL_NUM = {'AdvancedBarking':{'Barking':0,'Howling':1,'Crying':2},
                'HomeEmergency':{'HomeEmergency':0,'CO_Smoke_Alert':0},
                'GlassBreaking':{'GlassBreaking':0, 'Glass_Breaking':0},
                'JP_HomeEmergency':{'JP_HomeEmergency':0,'CO_Smoke_Alert':0,'JP_Alert':0,'HomeEmergency':0},
                'FCN':{'Cat_Meow':0,'Cat_Fighting':1,'Cat_Crying':2}}
OTHER_NUM = {'AdvancedBarking':3,'HomeEmergency':1,'GlassBreaking':1,'JP_HomeEmergency':1,'FCN':3}

def main():
    parser = ArgumentParser()
    # Arguments for convert label format
    parser.add_argument("-c","--csvfile", type=str, default="", help="csv data path")
    parser.add_argument("-s","--savecsv", type=str, default="")
    parser.add_argument("-t","--task", type=str, default="", choices=['AdvancedBarking', 'HomeEmergency', 'GlassBreaking', 'JP_HomeEmergency', 'FCN'], help="different training task")
    args = parser.parse_args()

    # training and val format
    savecsvfile = open(os.path.join(args.savecsv), 'w', newline='')
    writer = csv.writer(savecsvfile)
    writer.writerow(['path','label','remark'])

    print("convert to training or validation format ... ")
    data = pd.read_csv(args.csvfile) 
    for i in range(data.shape[0]):
        d = eval(data['remark'][i])
        if len(d) == 1 and d[0] in LABEL_NUM[args.task].keys():
            writer.writerow([data['path'][i], LABEL_NUM[args.task][d[0]], d])
        else:
            # filter out mulit label 
            choose_as_trainvalset = True
            for c in LABEL_NUM[args.task].keys():
                if c in d: choose_as_trainvalset = False
            
            if choose_as_trainvalset: writer.writerow([data['path'][i], OTHER_NUM[args.task], d])

    print("save a train val file with {} format in : {}".format(args.task,args.savecsv))

if __name__ == '__main__':
    main()
