# This module holds functions for crafting the benchmarking datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler

import argparse


def craft_credit():
    """
    This function loads credit dataset and:
        - creates PV flag column: 0 - majority, 1 - minority
        - renames 'default payment next month' with OUTLIER
        - drops ID column
        - preprocesses data: carries one hot encoding on discrete features and normalises continuous ones
    then saves crafted dataset in /datasets/proc folder.
    """
    # load dataset
    credit_df = pd.read_csv('datasets/raw/uci_credit.csv')

    # rename outlier column
    credit_df.rename(columns={"default payment next month": "OUTLIER"}, inplace=True)
    # drop ID column
    credit_df.drop(columns=['ID'], inplace=True)
    # PV flag: majority - age>25, minority - age<=25
    PV = credit_df.AGE.lt(25).astype(int)
    credit_df['PV'] = PV

    # Preprocess data
    dis_features = ['SEX','MARRIAGE', 'EDUCATION', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 'OUTLIER', 'PV']
    one_hot_features = [var for var in dis_features if not(var in ['OUTLIER', 'PV'])]
    cts_features = [var for var in credit_df.columns if not(var in dis_features)]

    # OneHotEncoding
    dummy = pd.get_dummies(credit_df[one_hot_features].astype(str))
    credit_df = pd.concat((credit_df,dummy), axis=1).drop(columns=one_hot_features+['SEX_2']) # concatenate & drop

    # Normalise
    credit_df[cts_features] = PowerTransformer().fit_transform(credit_df[cts_features])
    credit_df[cts_features] = MinMaxScaler(feature_range=(-1,1)).fit_transform(credit_df[cts_features])

    # save as csv
    credit_df.to_csv('datasets/proc/crafted_credit.csv', index=False)


def craft_adult():
    """
    This function loads adult dataset and:
        - OH encodes categorical variables
        - Binary encodes sex
        - Normalise continuous variables
        - Creates OUTLIER flag column
    then saves crafted dataset in /datasets/proc folder.
    """
    adult = pd.read_csv('datasets/raw/adult.csv')

    binary_features = ['sex']
    dis_features = ['workclass', 'education', 'marital-status',
                    'occupation', 'race', 'native-country', 'relationship']
    cont_features = [
        c for c in adult.columns if c not in dis_features and c != "label" and c not in binary_features]

    oh_encoded = pd.get_dummies(adult[dis_features])
    normalized_cont = PowerTransformer().fit_transform(adult[cont_features])
    normalized_cont = MinMaxScaler(feature_range=(-1,1)).fit_transform(normalized_cont)
    normalized_cont = pd.DataFrame(normalized_cont, columns=cont_features)

    oh_encoded['sex'] = 0
    oh_encoded.loc[adult['sex'] == ' Female', 'sex'] = 1

    adult_proc = pd.concat([oh_encoded, normalized_cont], axis=1)
    adult_proc['OUTLIER'] = adult.apply(lambda r: 1 if r['label'] == ' >50K' else 0, axis=1)

    # Subsample females to be in ratio 1:4 to males
    adult_males = adult_proc[adult_proc['sex'] == 0]
    adult_females = adult_proc[adult_proc['sex'] == 1].sample(len(adult_males) // 4)
    adult_proc = pd.concat([adult_males, adult_females], axis=0).sample(frac=1)

    # TODO: Further subsample women so outlier percentage is 5% for both male an
    # female

    # save as csv
    adult_proc.to_csv('datasets/proc/crafted_adult.csv', index=False)


def craft_insurance():
    insurance = pd.read_csv('datasets/raw/insurance.csv')

    # Costs are high over the 85th percentile (i.e. about 15% are anomalies)
    insurance['target'] = (insurance['charges'] > np.percentile(insurance.charges, 85)).astype(int)
    insurance = insurance.drop('charges', axis=1)
    insurance['sex'] = pd.get_dummies(insurance.sex)['male']
    insurance['smoker'] = pd.get_dummies(insurance.smoker)['yes']
    insurance['bmi'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(insurance['bmi'].values.reshape(-1,1))
    insurance['age'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(insurance['age'].values.reshape(-1,1))
    insurance['children'] = PowerTransformer().fit_transform(insurance['children'].values.reshape(-1,1))
    insurance = pd.concat([insurance, pd.get_dummies(insurance.region)], axis=1)
    insurance = insurance.drop("region", axis=1)

    # Subsample women as they are the minority group
    insurance_males = insurance[insurance.sex==1]
    insurance_females = insurance[insurance.sex==0].sample(frac=.1)
    insurance_subsampled = pd.concat([insurance_males, insurance_females]).sample(frac=1)

    insurance_subsampled.to_csv('datasets/proc/crafted_insurance.csv', index=False)

def craft_kdd():
    columns = ['AAGE', 'ACLSWKR', 'ADTIND', 'ADTOCC', 'AHGA', 'AHRSPAY',
                    'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN',
                    'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT', 'CAPGAIN', 'CAPLOSS',
                    'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX', 'HHDREL',
                    'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN',
                    'NOEMP', 'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP',
                    'SEOTR', 'VETQVA', 'VETYN', 'WKSWORK', 'YEAR',
                    'target']

    kdd = pd.read_csv('datasets/raw/kdd.csv', names=columns).dropna()

    # Remove NaN entries
    kdd = kdd[(kdd.GRINST != ' ?') & (kdd.HHDFMX != ' Grandchild <18 ever marr not in subfamily')]
    # Too high cardinality for common preproc techniques
    forbidden_columns = ['ADTIND', 'ADTOCC', 'SEOTR', 'VETYN', 'YEAR', "GRINST", "GRINREG"]
    # Countries. We will set National if "United States", else "foreigner"
    forbidden_columns += ['PEFNTVTY', 'PEMNTVTY', 'PENATVTY']

    continuous_columns = ['AAGE', 'AHRSPAY', 'DIVVAL', 'NOEMP', 'CAPGAIN', 'CAPLOSS', 'WKSWORK', 'MARSUPWT']
    binary_columns = ['ASEX', 'target']
    dummy_cols = [c for c in kdd.columns if c not in continuous_columns and c not in binary_columns and c != 'target']
    dummy_cols = [d for d in dummy_cols if d not in forbidden_columns]
    
    #Binarization
    kdd.target = kdd.apply(lambda r: 1 if r['target'] == ' 50000+.' else 0, axis=1) # .value_counts()
    kdd.ASEX = kdd.apply(lambda r: 1 if r['ASEX'] == ' Male' else 0, axis=1) # .value_counts()

    # convert multi-category to binary in order to reduce output dimensionality
    kdd["NATIONAL_FATHER"] = kdd.apply(lambda r: 1 if r['PEFNTVTY'] == ' United-States' else 0, axis=1) # .value_counts()
    kdd["NATIONAL_MOTHER"] = kdd.apply(lambda r: 1 if r['PEMNTVTY'] == ' United-States' else 0, axis=1) # .value_counts()
    kdd["NATIONAL_SELF"] = kdd.apply(lambda r: 1 if r['PENATVTY'] == ' United-States' else 0, axis=1) # .value_counts()
    # These new cols are now binary
    binary_columns += ["NATIONAL_FATHER", "NATIONAL_MOTHER", "NATIONAL_SELF"]

    dummy = pd.get_dummies(kdd[dummy_cols])
    cont = PowerTransformer().fit_transform(kdd[continuous_columns])
    cont = MinMaxScaler(feature_range=(-1,1)).fit_transform(cont)

    df_copy = kdd.copy().loc[:, continuous_columns]
    df_copy.loc[:, continuous_columns] = cont
    df_copy.loc[:, binary_columns] = kdd.loc[:, binary_columns]


    proc_df = pd.concat([df_copy, dummy], axis=1)
    
    proc_df_males = proc_df.loc[proc_df['ASEX']==0, :]
    proc_df_females = proc_df.loc[proc_df['ASEX']==1, :].sample(frac=.1)

    subsampled_df = pd.concat([proc_df_males, proc_df_females], axis=0).sample(frac=1)
    subsampled_df.to_csv('datasets/proc/crafted_kdd.csv', index=False)


def build_synth_dataset(mu_x, mu_o, sigma_x, sigma_o, num_points=5000, percent_outliers=.01, p=1/5):
    """
    This function generates a synthetic dataset: two features normally distributed with their respectives means and variances,
    target feature 'outlier' flag for 1% of the data points.

    Args:
        num_points (int): number of data points to generate
        percent_outliers (int): percentage of data points to label as outliers
        mu_x (double list): list of means for inlier features (number of features=2)
        mu_o (double list): list of means for outlier features (number of features=2)
        sigma_x (double list): list of variances for inlier features
        sigma_o (double list): list of variances for outlier features

    Return:
        dataframe: synthetic dataset
    """
    n_outliers = int(num_points * percent_outliers)

    x1, x2 = np.random.normal(mu_x[0], sigma_x[0], num_points-n_outliers),  np.random.normal(mu_x[1], sigma_x[1], num_points-n_outliers)
    o1, o2 = np.random.normal(mu_o[0], sigma_o[0], n_outliers),  np.random.normal(mu_o[1], sigma_o[1], n_outliers)

    inliers = np.array([x1,x2]).T
    outliers = np.array([o1,o2]).T

    y = np.ones(num_points)
    y[:num_points-n_outliers] = 0

    df = pd.DataFrame(
        np.vstack([inliers, outliers]),
        columns=['x1', 'x2']
    )

    df['outlier'] = y.astype(int)

    pv = np.random.binomial(1, p, num_points).T
    df['PV'] = pv

    df = df.sample(frac=1)

    return df



def main(dataset):
    avail_datasets = {
        'adult': craft_adult,
        'insurance': craft_insurance,
        'synth': build_synth_dataset,
        'credit': craft_credit,
        'kdd': craft_kdd
    }

    if dataset == 'all':
        for _, item in avail_datasets.items():
            item()
    else:
        avail_datasets[dataset]()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build dataset.')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset you want to build. "all" for building all of them',
                        choices=['all', 'adult', 'credit', 'insurance', 'kdd'])
    args = parser.parse_args()

    main(args.dataset[0])