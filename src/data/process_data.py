# This module holds functions for crafting the benchmarking datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, MinMaxScaler, StandardScaler, scale

import argparse


def naive_processing(data):
    cols = {}  
    xcat_cols = []
    dtypes = data.dtypes
    cols['cat'] = dtypes[dtypes == 'object'].index.to_list()
    cols['num'] = list(set(list(data.columns)).difference(set(cols['cat'])))

    cat_num = 0
    if len(cols['cat']) > 0:
        x_cat = pd.get_dummies(data[cols['cat']].astype('object'))
        xcat_cols = x_cat.columns.tolist()
        x_cat = x_cat.values
        cat_num = x_cat.shape[1]

    num_num = 0
    if len(cols['num']) > 0:
        x_num = scale(data[cols['num']])
        num_num = x_num.shape[1]

    if num_num > 0 and cat_num > 0:
        x = np.hstack((x_num, x_cat))
    elif num_num > 0:
        x = x_num
    elif cat_num > 0:
        x = x_cat
    
    return pd.DataFrame(x, columns=cols['num'] + xcat_cols)



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

    credit_df = credit_df.rename(columns={"default payment next month": "OUTLIER"})
    credit_df.drop('ID', axis=1, inplace=True)

    # Some cols have negative values. This could mean something (maybe returned money), but, 
    # as it's not specified, they will be clipped to 0 (as those high negative values will hurt the NNs)
    bill_colnames = ['BILL_AMT{}'.format(i) for i in range(1, 7)]
    credit_df.loc[:, bill_colnames] = np.maximum(0, credit_df[bill_colnames]) 

    # TODO: Remove high cardinality columns
    #high_car_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    #credit_df.drop(high_car_cols, axis=1, inplace=True)
    #dis_features = ['SEX', 'OUTLIER', 'MARRIAGE', 'EDUCATION']

    # TODO: Remove 0 entries so the norm. distribution is not bimodal
    for c in [f'PAY_AMT{i}' for i in range(1, 7)]:
        credit_df.loc[credit_df[c] <= 0, c] = credit_df[c].median()
    for c in [f'BILL_AMT{i}' for i in range(1, 7)]:
        credit_df.loc[credit_df[c] <= 0, c] = credit_df[c].median()

    dis_features = ['SEX', 'OUTLIER', 'MARRIAGE', 'EDUCATION', 'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

    one_hot_features = [var for var in dis_features if not(var in ['OUTLIER', 'SEX'])]
    cts_features = [var for var in credit_df.columns if not(var in dis_features)]


    #dummy = pd.get_dummies(credit_df[one_hot_features].astype(str))
    #cts = credit_df.loc[:, cts_features]

    #credit_df_proc = pd.concat((cts,dummy), axis=1)
    credit_df_proc = credit_df
    credit_df_proc['OUTLIER'] = credit_df['OUTLIER']
    credit_df_proc['SEX'] = credit_df['SEX'] - 1


    credit_df_proc_men = credit_df_proc[credit_df_proc['SEX'] == 1]
    credit_df_proc_women = credit_df_proc[credit_df_proc['SEX'] == 0]

    credit_df_proc_subsampled = pd.concat([
        credit_df_proc_men,
        credit_df_proc_women.sample(int(len(credit_df_proc_men) * 0.25))
    ], axis=0).sample(frac=1)

    credit_df_proc_inliers = credit_df_proc_subsampled[credit_df_proc_subsampled['OUTLIER'] == 0]
    credit_df_proc_outliers = credit_df_proc_subsampled[credit_df_proc_subsampled['OUTLIER'] == 1]

    credit_df_proc_subsampled = pd.concat([
        credit_df_proc_inliers,
        credit_df_proc_outliers.sample(int(len(credit_df_proc_outliers) * 0.1))
    ], axis=0).sample(frac=1).reset_index(drop=True)

    # Standarize cont feats
    #normalized_cont = credit_df_proc_subsampled[cts_features]
    #normalized_cont = PowerTransformer().fit_transform(normalized_cont)
    #normalized_cont = StandardScaler().fit_transform(normalized_cont)
    #credit_df_proc_subsampled.loc[:, cts_features] = normalized_cont
    pv = credit_df_proc_subsampled['SEX'].reset_index(drop=True)
    tgt = credit_df_proc_subsampled['OUTLIER'].reset_index(drop=True)

    credit_df_proc_subsampled = naive_processing(credit_df_proc_subsampled.drop(['SEX', 'OUTLIER'], axis=1))
    credit_df_proc_subsampled['SEX'] = pv.astype(int)
    credit_df_proc_subsampled['OUTLIER'] = tgt.astype(int)

    # import pdb; pdb.set_trace()

    # save as csv
    credit_df_proc_subsampled.to_csv('datasets/proc/crafted_credit.csv', index=False)


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

    # TODO: Only two cats for this
    mask = adult['native-country'] == ' United-States'
    adult.loc[mask, 'native-country'] = 'national'
    adult.loc[~mask, 'native-country'] = 'non-national'

    oh_encoded = pd.get_dummies(adult[dis_features])
    #normalized_cont = PowerTransformer().fit_transform(adult[cont_features])
    #normalized_cont = MinMaxScaler(feature_range=(-1,1)).fit_transform(normalized_cont)
    #normalized_cont = StandardScaler().fit_transform(normalized_cont)
    
    normalized_cont = StandardScaler().fit_transform(adult[cont_features])
    
    normalized_cont = pd.DataFrame(normalized_cont, columns=cont_features)

    oh_encoded['sex'] = 0
    oh_encoded.loc[adult['sex'] == ' Female', 'sex'] = 1

    adult_proc = pd.concat([oh_encoded, normalized_cont], axis=1)
    adult_proc['OUTLIER'] = adult.apply(lambda r: 1 if r['label'] == ' >50K' else 0, axis=1)

    # Subsample females to be in ratio 1:4 to males
    #adult_males = adult_proc[adult_proc['sex'] == 0]
    #adult_females = adult_proc[adult_proc['sex'] == 1].sample(len(adult_males) // 4)
    #adult_proc = pd.concat([adult_males, adult_females], axis=0).sample(frac=1)

    # TODO: Further subsample women so outlier percentage is 5% for both male an
    # female

    # save as csv
    adult_proc.to_csv('datasets/proc/crafted_adult.csv', index=False)


def craft_insurance():
    insurance = pd.read_csv('datasets/raw/insurance.csv')
    
    tgt = (insurance['charges'] > np.percentile(insurance.charges, 85)).astype(int)
    sex = pd.get_dummies(insurance.sex)['male']

    encoded_df = naive_processing(insurance.drop(['charges', 'sex'], axis=1))
    encoded_df['target'] = tgt
    encoded_df['sex'] = sex
    insurance = encoded_df

    # # Costs are high over the 85th percentile (i.e. about 15% are anomalies)
    # insurance['target'] = (insurance['charges'] > np.percentile(insurance.charges, 85)).astype(int)
    # insurance = insurance.drop('charges', axis=1)
    # insurance['sex'] = pd.get_dummies(insurance.sex)['male']
    # insurance['smoker'] = pd.get_dummies(insurance.smoker)['yes']
    # # insurance['bmi'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(insurance['bmi'].values.reshape(-1,1))
    # # insurance['age'] = MinMaxScaler(feature_range=(-1,1)).fit_transform(insurance['age'].values.reshape(-1,1))
    # insurance['bmi'] = StandardScaler().fit_transform(insurance['bmi'].values.reshape(-1,1))
    # insurance['age'] = StandardScaler().fit_transform(insurance['age'].values.reshape(-1,1))
    # insurance['children'] = PowerTransformer().fit_transform(insurance['children'].values.reshape(-1,1))
    # insurance = pd.concat([insurance, pd.get_dummies(insurance.region)], axis=1)
    # insurance = insurance.drop("region", axis=1)

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
    #cont = MinMaxScaler(feature_range=(-1,1)).fit_transform(cont)
    cont = StandardScaler().fit_transform(cont)

    df_copy = kdd.copy().loc[:, continuous_columns]
    df_copy.loc[:, continuous_columns] = cont
    df_copy.loc[:, binary_columns] = kdd.loc[:, binary_columns]


    proc_df = pd.concat([df_copy, dummy], axis=1)
    
    proc_df_males = proc_df.loc[proc_df['ASEX']==0, :]
    proc_df_females = proc_df.loc[proc_df['ASEX']==1, :].sample(frac=.1)

    subsampled_df = pd.concat([proc_df_males, proc_df_females], axis=0).sample(frac=1)
    subsampled_df.to_csv('datasets/proc/crafted_kdd.csv', index=False)

def craft_bank():
    dataset = pd.read_csv('datasets/raw/bank.csv', sep=';')

    age = (dataset.age < 25).astype(int)
    label = (dataset.y == 'yes').astype(int)

    proc_data = naive_processing(
        dataset.drop(['age', 'y'], axis=1)
    )
    proc_data['age'] = age
    proc_data['label'] = label
    proc_data.to_csv('datasets/proc/crafted_bank.csv', index=False)

def craft_obesity():
    dataset = pd.read_csv('datasets/raw/obesity.csv')
    dataset['target'] = dataset.apply(lambda r: 1 if r['NObeyesdad'] == 'Insufficient_Weight' else 0, axis=1)
    gender = (dataset.Gender == 'Female').astype(int)
    tgt = dataset['target']
    dataset.drop(['NObeyesdad', 'Gender', 'target'], axis=1, inplace=True)

    dataset_mod = naive_processing(dataset)
    dataset_mod['Gender'] = gender
    dataset_mod['target'] = tgt


    # dataset_mod = dataset.copy()

    # dataset_mod['Gender'] = dataset_mod.apply(lambda r: 1 if r['Gender'] == 'Female' else 0, axis=1)
    # dataset_mod['family_history_with_overweight'] = dataset_mod.apply(lambda r: 1 if r['family_history_with_overweight'] == 'yes' else 0, axis=1)
    # dataset_mod['SCC'] = dataset_mod.apply(lambda r: 1 if r['SCC'] == 'yes' else 0, axis=1)
    # dataset_mod['FAVC'] = dataset_mod.apply(lambda r: 1 if r['FAVC'] == 'yes' else 0, axis=1)
    # dataset_mod['SMOKE'] = dataset_mod.apply(lambda r: 1 if r['SMOKE'] == 'yes' else 0, axis=1)
    # dataset_mod['target'] = dataset_mod.apply(lambda r: 1 if r['NObeyesdad'] == 'Insufficient_Weight' else 0, axis=1)

    # # water liters drunk
    # mask_3l = dataset_mod.CH2O >= 3
    # mask_23l = (dataset_mod.CH2O > 2) & (dataset_mod.CH2O < 3)
    # mask_12l = (dataset_mod.CH2O > 1) & (dataset_mod.CH2O <= 2)
    # mask_1l = (dataset_mod.CH2O <= 1)

    # dataset_mod.loc[mask_3l, 'CH2O'] = '3L'
    # dataset_mod.loc[mask_23l, 'CH2O'] = '2-3L'
    # dataset_mod.loc[mask_12l, 'CH2O'] = '1-2L'
    # dataset_mod.loc[mask_1l, 'CH2O'] = '0-1L'

    # dataset_mod['FAF'] = dataset_mod.FAF.astype(int).astype(str) + 'l' #physical activity freq
    # dataset_mod['TUE'] = dataset_mod.TUE.astype(int).astype(str) + 'l' # usage of tech devices
    # dataset_mod['NCP'] = dataset_mod.NCP.astype(int).astype(str) + 'l' # caloric foods eaten
    # dataset_mod['FCVC'] = dataset_mod.FCVC.astype(int).astype(str) + 'l' # vegetables eaten

    # dummycols = ['CAEC', 'CALC', 'MTRANS', 'CH2O', 'FAF', 'TUE', 'NCP', 'FCVC']
    # dummies = pd.get_dummies(dataset_mod.loc[:, dummycols])

    # dataset_mod = dataset_mod.drop(dummycols, axis=1)
    # dataset_mod = dataset_mod.drop('NObeyesdad', axis=1)

    # dataset_mod = pd.concat([dataset_mod, dummies], axis=1)
    # scaled_cont = PowerTransformer().fit_transform(dataset_mod.loc[:, ['Height', 'Weight', 'Age']])
    # scaled_cont = StandardScaler().fit_transform(scaled_cont)

    # dataset_mod.loc[:, ['Height', 'Weight', 'Age']] = scaled_cont

    # Subsample women (to about 8%)
    dataset_males = dataset_mod[dataset_mod['Gender'] == 0]
    dataset_females = dataset_mod[dataset_mod['Gender'] == 1].sample(frac=.1)
    dataset_resampled = pd.concat([dataset_males, dataset_females], axis=0).sample(frac=1)

    dataset_resampled.to_csv('datasets/proc/crafted_obesity.csv', index=False)


def craft_celeba():
    dataset = pd.read_csv('datasets/raw/celeba.csv')
    dataset.drop(columns='image_id', inplace=True)

    # Change flags:  -1 -> 0
    for c in dataset.columns: dataset.loc[dataset[c] == -1, c] = 0

    # Subsample men (to 10%)
    males = dataset[dataset['Male'] == 1].sample(frac=.1)
    females = dataset[dataset.Male == 0]
    dataset_subsampled = pd.concat([males, females]).sample(frac=1)

    # Subsample attractive people (outliers) to about 5%
    not_attractive = dataset_subsampled[dataset_subsampled.Attractive == 0].sample(frac=.05)
    attractive = dataset_subsampled[dataset_subsampled.Attractive == 1]
    dataset_subsampled = pd.concat([attractive, not_attractive]).sample(frac=1)

    dataset_subsampled.to_csv('datasets/proc/crafted_celeba.csv', index=False)


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
        'bank': craft_bank,
        'insurance': craft_insurance,
        'credit': craft_credit,
        'kdd': craft_kdd,
        'obesity': craft_obesity,
        'celeba': craft_celeba
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
                        choices=['all', 'adult', 'bank', 'credit', 'insurance', 'kdd', 'obesity', 'celeba'])
    args = parser.parse_args()

    main(args.dataset[0])