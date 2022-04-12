import os
import csv 

import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser


def main():
    # Arguments for choosing data
    parser = ArgumentParser()
    parser.add_argument("-c","--csvfile", type=str, nargs='+', default="", help="csv data path")
    parser.add_argument("-s","--savecsv", type=str, default="", help="save csv path")
    args = parser.parse_args()

    # training and val format
    savecsvfile = open(os.path.join(args.savecsv), 'w', newline='')
    writer = csv.writer(savecsvfile)
    writer.writerow(['path','label'])


    for f in args.csvfile:
        data = pd.read_csv(f)
        for i in range(data.shape[0]):
            writer.writerow([data['path'][i], data['label'][i]])
    
if __name__ == '__main__':
    main()
