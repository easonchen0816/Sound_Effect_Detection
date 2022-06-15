from sklearn.model_selection import train_test_split
import pandas as pd

from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("-c","--csvfile", type=str, default="", help="csv data path")
    parser.add_argument("-t","--traincsv", type=str, default="", help="training csv output data path")
    parser.add_argument("-v","--valcsv", type=str, default="", help="validation csv output data path")
    parser.add_argument("-r","--valratio", type=float, default="", help="validation ratio")
    args = parser.parse_args()

    print("Reading data from {} ...".format(args.csvfile))
    data = pd.read_csv(args.csvfile)
    print("Splitting data with validation ratio {} ...".format(args.valratio))
    train, val = train_test_split(data, test_size=args.valratio)
    print("Saving train/val data...")
    train.to_csv(args.traincsv,index=None)
    val.to_csv(args.valcsv,index=None)

if __name__ == "__main__":
    main()