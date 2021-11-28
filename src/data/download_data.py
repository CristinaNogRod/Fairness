import pandas as pd
import argparse
import requests
import tempfile
import requests
import gzip
import shutil

# Download the UCI Credit Dataset
def download_credit():
    print("UCI Credit dataset")
    print("\t More info: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
    print('Downloading UCI Credit Dataset to datasets/raw/uci_credit.csv...')
    credit = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls', header=1)
    # Save it inside the data folder as CSV
    credit.to_csv('datasets/raw/uci_credit.csv', index=False)
    print("Done downloading UCI Credit!\n\n")

def download_adult():
    print("Adult dataset")
    print("\t More info: https://archive.ics.uci.edu/ml/datasets/adult")
    print("Downloading Adult dataset to datasets/raw/adult.csv...")
    adult_cols = ["age", "workclass", "fnlwgt", "education",
                "education-num", "marital-status",
                "occupation", "relationship", "race",
                "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country", "label"]
    adult = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=0, names=adult_cols, index_col=False)
    adult.to_csv('datasets/raw/adult.csv', index=False)
    print("Done downloading Adult dataset!\n\n")

def download_insurance():
    print("Insurance dataset")
    print("\t More info: https://www.kaggle.com/mirichoi0218/insurance")
    print("Downloading Insurance dataset to datasets/raw/insurance.csv...")
    insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    insurance.to_csv('datasets/raw/insurance.csv', index=False)
    print("Done downloading Insurance dataset!\n\n")

def download_census():
    print("KDD U.S Census dataset")
    print("\t More info: https://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29")
    print("Downloading KDD U.S Census dataset to datasets/raw/kdd.csv...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census-income-mld/census-income.data.gz"
    path_out = "datasets/raw/kdd.csv"

    with tempfile.TemporaryDirectory() as tmpdirname:
        target_path = tmpdirname + "/census.tar.gz"

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                f.write(response.raw.read())
        
        with gzip.open(target_path, 'rb') as f_in:
            with open(path_out, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    print("Done downloading Insurance dataset!\n\n")

# TODO: Add rest of the dataset downloading here


def main(dataset):
    avail_datasets = {
        'adult': download_adult,
        'insurance': download_insurance,
        'credit': download_credit,
        'kdd': download_census
    }

    if dataset == 'all':
        for _, item in avail_datasets.items():
            item()
    else:
        avail_datasets[dataset]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download dataset.')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset you want to download. "all" for downloading all of them',
                        choices=['all', 'adult', 'credit', 'insurance', 'kdd'])
    args = parser.parse_args()

    main(args.dataset[0])