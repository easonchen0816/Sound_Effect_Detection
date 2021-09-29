import os
import csv
import glob
import shutil

from argparse import ArgumentParser


def load_csv(csvpath):
    # Read csv file
    csvfile = open(os.path.join(csvpath))
    data = csv.reader(csvfile)
    return data

def main():
    parser = ArgumentParser()
    parser.add_argument("--org_path", type=str, help="audio data path in dvc. ex: /KIKI/hucheng/mini_advancedbarking_data/train/")
    parser.add_argument("--add_path", type=str, help="newly collected audio folder. ex: /KIKI/henry/add_mini_advancedbarking_data/train/")
    parser.add_argument("--org_csv" , type=str, help="the label csv files in dvc. ex: /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_01_train.csv")
    parser.add_argument("--add_csv" , type=str, help="newly collected label csv files. ex: /KIKI/henry/add_mini_advancedbarking_data/meta/add.csv")
    parser.add_argument("--new_csv" , type=str, help="create a new label csv file in dvc. Format: Task_Device_YYYYMMDD_Version_Type.csv. ex: /KIKI/hucheng/mini_advancedbarking_data/meta/AdvancedBarking_MC_20210805_02_train.csv")
    args = parser.parse_args()

    ######################
    # Integrate csv file #
    ######################
    # Create new csv file
    print("Create new label file here : {}".format(args.new_csv))
    savecsvfile = open(os.path.join(args.new_csv), 'w', newline='')
    writer = csv.writer(savecsvfile)

    # Load origin label csv in dvc and add csv file
    print("Load origin csv label file : {}".format(args.org_csv))
    print("Load add csv label file : {}".format(args.add_csv))
    origin_data = load_csv(args.org_csv)
    add_data = load_csv(args.add_csv)

    # Write in new csv
    print("Start to integrate label csv file...")
    origin_label = {} ; add_label = {} ; total_label = {}
    data_in_dvc_folder = [] ; data_in_add_folder = []

    for d in origin_data: 
        writer.writerow(d)
        data_in_dvc_folder.append(d[0])
        # Calculate origin label
        if d[0] != "path":
            if d[1] not in origin_label: origin_label[d[1]] = 1
            else: origin_label[d[1]] += 1

    for d in add_data: 
        if d[0] not in data_in_dvc_folder: 
            writer.writerow(d)
            data_in_add_folder.append(d[0])
            # Calculate add label
            if d[1] not in add_label: add_label[d[1]] = 1
            else: add_label[d[1]] += 1

    ##########################
    # Copy audio file to dvc #
    ##########################
    # Copy audio folder according to add_csv file
    print("Start to copy add audio files to dvc... (from {} to {})".format(args.add_path, args.org_path))
    add_data = load_csv(args.add_csv)
    for d in add_data: 
        dst = os.path.join(args.org_path, d[0])
        if d[0] not in data_in_dvc_folder: shutil.copyfile(os.path.join(args.add_path, d[0]), dst)

    #####################
    # Final data result #
    #####################
    print("Origin data quantity : {}".format(len(data_in_dvc_folder)-1))
    print("Add data quantity : {}".format(len(data_in_add_folder)))
    print("Final data quantity : {}".format(len(data_in_add_folder) + len(data_in_dvc_folder)-1))
    print("Origin label quantity :", origin_label)
    print("Add label quantity :", add_label)

if __name__ == "__main__":
    main()