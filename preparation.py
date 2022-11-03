import os
import csv
import numpy as np

def main():
    parser = ArgumentParser()
    # data or model path setting
    parser.add_argument("--sourcepath", default = "/home/henry/tomofun_modules/all_device_data/meta_furbo3_data.csv", type=str)    
    parser.add_argument("--targetpath", default = '/home/easonchen/sound_automation_retrain/Data/training_set.csv', type=str)
    parser.add_argument("--num_data", type=int)

    args = parser.parse_args()
    with open(args.targetpath, 'w', newline='') as out:
        with open(args.sourcepath, newline='') as f:
            count = 0
            rows = csv.reader(f)
            w = csv.writer(out)
            for row in rows:
                label = np.zeros(4)
                wavpath = row[0]
                old_label = row[1]
                old_label_len = len(row[1].split(','))
                # print('Barking' in old_label)
                if 'Barking' in old_label:
                    label[0] = 1
                if 'Howling' in old_label:
                    label[1] = 1
                if 'Crying' in old_label:
                    label[2] = 1
                if sum(label) != old_label_len:
                    # print(label, old_label)
                    label[3] = 1
                # elif len(label) == 0:
                #     print(label)
                #     label[3] = 1
                # if len(label) !=0 and label[0] == ',': 
                #     label = label[1:]
                if wavpath == 'path':
                    w.writerow([wavpath, 'label'])
                else:
                    w.writerow([wavpath, label])
                # print(row, label)
                count += 1
                if (count>=args.num_data):
                    break
        
