import os
import csv 

import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def main():
    # Arguments for choosing data
    parser = ArgumentParser()
    parser.add_argument("-d","--datacsv", type=str, default="", help="csv data path")
    parser.add_argument("-l","--label", type=str, nargs='+', default="", help="target label")
    parser.add_argument("-n","--num", type=str, nargs='+', default="", help="each label num")
    parser.add_argument("-t","--traintestset", type=str, nargs='+', default="", help="traintesting set csv")
    parser.add_argument("-c","--choosesingle", action="store_true", help="choose single label setting")
    parser.add_argument("-s","--savecsv", type=str, default="", help="save addition data csv path")
    args = parser.parse_args()

    # new data csvfile
    # open the file in the write mode
    f = open(args.savecsv, 'w')
    writer = csv.writer(f)
    writer.writerow(['path','remark'])

    # all testing data (new training data should not in testing set)
    testing_data = []
    for t in args.traintestset:
        t_data = pd.read_csv(t)
        for i in range(t_data.shape[0]): testing_data.append(t_data['path'][i])

    # target data and num
    target = {}
    choosing_num = {}
    for i in range(len(args.label)):
        target[args.label[i]] = int(args.num[i])
        choosing_num[args.label[i]] = 0
    
    # start to choose data
    data = pd.read_csv(args.datacsv)
    for i in tqdm(range(data.shape[0])):
        save_this_data = False
        path = data['path'][i]
        annotation = eval(data['remark'][i])
        
        # check whether should i choose this data
        # 1. data not in testing set
        # 2. data label is what we need
        if path not in testing_data:
            for a in annotation: 
                if a in target and target[a] > choosing_num[a]: 
                    save_this_data = True
        
        # single label filter
        if args.choosesingle:
            if len(annotation) != 1: save_this_data = False

        # save data
        if save_this_data:
            writer.writerow([path, annotation])
            for a in annotation: 
                if a in choosing_num: choosing_num[a] += 1
                else: choosing_num[a] = 1

    print("save new data in: ", args.savecsv)
    print("Here is the label we choosing: ", choosing_num)

if __name__ == '__main__':
    main()