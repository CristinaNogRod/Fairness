import pandas as pd


# Download the UCI Credit Dataset
print("UCI Credit dataset")
print("\t More info: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
print('Downloading UCI Credit Dataset to datasets/raw/uci_credit.csv...')
credit = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls', header=1)
# Save it inside the data folder as CSV
credit.to_csv('datasets/raw/uci_credit.csv', index=False)
print("Done downloading UCI Credit!\n\n")

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
print("Done downloading UCI Credit!\n\n")

# TODO: Add rest of the dataset downloading here
