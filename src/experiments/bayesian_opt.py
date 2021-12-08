import pandas as pd
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import argparse
import inspect
import sys
import os
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from fairsgod.losses import FairODLoss
from fairsgod.fairod import OutlierDetector
from fairsgod.fairod_sg import SGOutlierDetector
from fairsgod import evaluation
from sklearn.metrics import roc_auc_score

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Dummy wrapper bc scikit-optimize only deals with X, y (not PV)
class BayesWrapper():
    def __init__(self, *args, **kwargs):
        self. model = SGOutlierDetector(*args, **kwargs)
        
    def fit(self, X, y, batch_size=1024, epochs=100, val_X=None, val_pv=None, early_stop=True):
        pv = X.loc[:, 'pv']
        X = X.drop('pv', axis=1)
        self.model.fit(X, pv, batch_size, epochs, verbose=False, early_stop=early_stop)
        
    def score(self, X, y):
        y_pred =  self.model.predict_scores(X.drop('pv', axis=1))
        return roc_auc_score(y, y_pred)
    
    def get_params(self, *args, **kwargs):
        return self.model.get_params(*args, **kwargs)

    def set_params(self, **params):
        self.model = self.model.set_params(**params)
        return self


def run_optim(X_train, y_train, pv_train, bayesian_runs=32, seed=33, epsilon_p=90):
    opt = BayesSearchCV(
        BayesWrapper(),#SGOutlierDetector(epsilon=1e-6),#, alpha=0.5, gamma=0.5),
        {
            #'epsilon': Real(1e-6, 0.9, prior='log-uniform'), # 0.01, 
            'lambda_se': Real(.01, 100, prior='log-uniform'),
            'a': Real(5, 20, prior='log-uniform'),
            'lambda_a': Real(15.0, 100.0, prior='log-uniform'),
            'alpha': Real(.1, .6, prior='log-uniform'),
            'gamma': Real(.001, .8, prior='log-uniform')
        },
        n_iter=bayesian_runs,
        random_state=seed,
        fit_params= dict(
              batch_size=1024, epochs=100, val_X=None, val_pv=None, epsilon_p=epsilon_p,
        ),
        verbose=1,
        cv=2
    )

    optim_train_x = X_train.copy()
    optim_train_x['pv'] = pv_train
    _ = opt.fit(optim_train_x, y_train)

    best_params = dict(opt.best_params_)
    best_score_train = opt.best_score_

    return best_params, best_score_train

def run_eval(X_train, pv_train, X_test, pv_test, y_test, hyperparams, epochs=6):
    print()
    print("Training with optimum parameters and evaluating on TEST Set...")
    model = SGOutlierDetector(**hyperparams)
    model.fit(X_train, pv_train, batch_size=1024, epochs=epochs, verbose=False, early_stop=True)
    X_pred = model.predict_scores(X_test).numpy()

    fsg_od_metrics = {
        'auc': roc_auc_score(y_test, X_pred).astype(float),
        'auc_ratio': evaluation.auc_ratio(y_test, X_pred, pv_test).astype(float),
        'ap_ratio': evaluation.compute_AP_ratio(y_test, X_pred, pv_test).astype(float),
        'precision_ratio': evaluation.compute_precision_ratio(y_test, X_pred, pv_test).astype(float),
        'fairness': evaluation.compute_Fairness_metric(y_test, X_pred, pv_test).numpy().astype(float),
        'group_fidelity': evaluation.compute_GF_metric(X_pred, pv_test).numpy().astype(float)
    }

    return fsg_od_metrics


def opt_adult(seed=33, 
              datasets_base='datasets',
              bayesian_runs=32,
              report_path='outputs/optimizations'):
    """
    This func. runs a bayesian optimization over all hyperparams for the ADULT DATASET
    """
    print("Running Bayesian optimization for the ADULT dataset...")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    dataname = "adult"

    adult_df = pd.read_csv(f"{datasets_base}/proc/crafted_adult.csv")

    y_pv = adult_df[['OUTLIER','sex']]
    X = adult_df.drop(columns=['OUTLIER', 'sex'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True, random_state=seed)

    pv_test = y_pv_test['sex']
    pv_train = y_pv_train['sex']
    y_train = y_pv_train['OUTLIER']
    y_test = y_pv_test['OUTLIER']

    best_params, best_score_train = run_optim(X_train, y_train, pv_train, bayesian_runs=bayesian_runs, seed=seed)

    print("Optimization done")

    print(
        tabulate(
            pd.DataFrame.from_dict(best_params, orient='index')
        )
    )

    print(f"score:  {best_score_train.astype(float)}")
    print()

    fsg_od_metrics = run_eval(X_train, pv_train, X_test, pv_test, y_test, best_params)

    report_table = tabulate([
        ['AUC', fsg_od_metrics['auc']],
        ['AUC Ratio', fsg_od_metrics['auc_ratio']],
        ['AP Ratio', fsg_od_metrics['ap_ratio']],
        ['Precision Ratio', fsg_od_metrics['precision_ratio']],
        ['Fairness', fsg_od_metrics['fairness']],
        ['Group Fidelity', fsg_od_metrics['group_fidelity']],
    ], headers=['SG-FOD'])

    print("Metrics report...")
    print(report_table)
    print()

    optimization_report = {
        'test_metrics': fsg_od_metrics,
        'optim_best_score': best_score_train,
        'optim_best_params': best_params
    }

    with open(report_path + f"/optim_{dataname}.json", 'w') as file:
        json.dump(optimization_report, file, indent=4)

def opt_credit(seed=33, 
              datasets_base='datasets',
              bayesian_runs=32,
              report_path='outputs/optimizations'):

    dataname = "credit"

    print("Running Bayesian optimization for the CREDIT dataset...")
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # load credit
    credit_df = pd.read_csv(f"{datasets_base}/proc/crafted_credit.csv")

    y_pv = credit_df[['OUTLIER','SEX']]
    X = credit_df.drop(columns=['OUTLIER', 'AGE', 'SEX'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True, random_state=seed)

    pv_test = y_pv_test['SEX']
    pv_train = y_pv_train['SEX']
    y_train = y_pv_train['OUTLIER']
    y_test = y_pv_test['OUTLIER']

    best_params, best_score_train = run_optim(X_train, y_train, pv_train, bayesian_runs=bayesian_runs, seed=seed)

    print("Optimization done")

    print(
        tabulate(
            pd.DataFrame.from_dict(best_params, orient='index')
        )
    )

    print(f"score:  {best_score_train.astype(float)}")
    print()

    fsg_od_metrics = run_eval(X_train, pv_train, X_test, pv_test, y_test, best_params)

    report_table = tabulate([
        ['AUC', fsg_od_metrics['auc']],
        ['AUC Ratio', fsg_od_metrics['auc_ratio']],
        ['AP Ratio', fsg_od_metrics['ap_ratio']],
        ['Precision Ratio', fsg_od_metrics['precision_ratio']],
        ['Fairness', fsg_od_metrics['fairness']],
        ['Group Fidelity', fsg_od_metrics['group_fidelity']],
    ], headers=['SG-FOD'])

    print("Metrics report...")
    print(report_table)
    print()

    optimization_report = {
        'test_metrics': fsg_od_metrics,
        'optim_best_score': best_score_train,
        'optim_best_params': best_params
    }

    with open(report_path + f"/optim_{dataname}.json", 'w') as file:
        json.dump(optimization_report, file, indent=4)

def opt_kdd(seed=33, 
              datasets_base='datasets',
              bayesian_runs=32,
              report_path='outputs/optimizations'):

    dataname = "kdd"

    print("Running Bayesian optimization for the KDD dataset...")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    kdd_df = pd.read_csv(f"{datasets_base}/proc/crafted_kdd.csv")

    y_pv = kdd_df[['target','ASEX']]
    X = kdd_df.drop(columns=['target', 'ASEX'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True, random_state=seed)

    pv_test = y_pv_test['ASEX']
    pv_train = y_pv_train['ASEX']
    y_train = y_pv_train['target']
    y_test = y_pv_test['target']

    best_params, best_score_train = run_optim(X_train, y_train, pv_train, bayesian_runs=bayesian_runs, seed=seed)

    print("Optimization done")

    print(
        tabulate(
            pd.DataFrame.from_dict(best_params, orient='index')
        )
    )

    print(f"score:  {best_score_train.astype(float)}")
    print()

    fsg_od_metrics = run_eval(X_train, pv_train, X_test, pv_test, y_test, best_params)

    report_table = tabulate([
        ['AUC', fsg_od_metrics['auc']],
        ['AUC Ratio', fsg_od_metrics['auc_ratio']],
        ['AP Ratio', fsg_od_metrics['ap_ratio']],
        ['Precision Ratio', fsg_od_metrics['precision_ratio']],
        ['Fairness', fsg_od_metrics['fairness']],
        ['Group Fidelity', fsg_od_metrics['group_fidelity']],
    ], headers=['SG-FOD'])

    print("Metrics report...")
    print(report_table)
    print()

    optimization_report = {
        'test_metrics': fsg_od_metrics,
        'optim_best_score': best_score_train,
        'optim_best_params': best_params
    }

    with open(report_path + f"/optim_{dataname}.json", 'w') as file:
        json.dump(optimization_report, file, indent=4)

def opt_obesity(seed=33, 
              datasets_base='datasets',
              bayesian_runs=32,
              report_path='outputs/optimizations'):

    dataname = "obesity"

    print("Running Bayesian optimization for the Obesity dataset...")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    obesity_df = pd.read_csv(f"{datasets_base}/proc/crafted_obesity.csv")

    y_pv = obesity_df[['target','Gender']]
    X = obesity_df.drop(columns=['target', 'Gender'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True, random_state=seed)

    pv_test = y_pv_test['Gender']
    pv_train = y_pv_train['Gender']
    y_train = y_pv_train['target']
    y_test = y_pv_test['target']

    best_params, best_score_train = run_optim(X_train, y_train, pv_train, bayesian_runs=bayesian_runs, seed=seed)

    print("Optimization done")

    print(
        tabulate(
            pd.DataFrame.from_dict(best_params, orient='index')
        )
    )

    print(f"score:  {best_score_train.astype(float)}")
    print()

    fsg_od_metrics = run_eval(X_train, pv_train, X_test, pv_test, y_test, best_params)

    report_table = tabulate([
        ['AUC', fsg_od_metrics['auc']],
        ['AUC Ratio', fsg_od_metrics['auc_ratio']],
        ['AP Ratio', fsg_od_metrics['ap_ratio']],
        ['Precision Ratio', fsg_od_metrics['precision_ratio']],
        ['Fairness', fsg_od_metrics['fairness']],
        ['Group Fidelity', fsg_od_metrics['group_fidelity']],
    ], headers=['SG-FOD'])

    print("Metrics report...")
    print(report_table)
    print()

    optimization_report = {
        'test_metrics': fsg_od_metrics,
        'optim_best_score': best_score_train,
        'optim_best_params': best_params
    }

    with open(report_path + f"/optim_{dataname}.json", 'w') as file:
        json.dump(optimization_report, file, indent=4)


def opt_bank(seed=33, 
              datasets_base='datasets',
              bayesian_runs=32,
              report_path='outputs/optimizations'):

    dataname = "bank"

    print("Running Bayesian optimization for the Bank dataset...")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    bank_df = pd.read_csv(f"{datasets_base}/proc/crafted_bank.csv")
    y_pv = bank_df[['label','age']]
    X = bank_df.drop(columns=['label', 'age'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True)

    pv_test = y_pv_test['age']
    pv_train = y_pv_train['age']
    y_test = y_pv_test['label']
    y_train = y_pv_train['label']

    best_params, best_score_train = run_optim(X_train, y_train, pv_train, bayesian_runs=bayesian_runs, seed=seed)

    print("Optimization done")

    print(
        tabulate(
            pd.DataFrame.from_dict(best_params, orient='index')
        )
    )

    print(f"score:  {best_score_train.astype(float)}")
    print()

    fsg_od_metrics = run_eval(X_train, pv_train, X_test, pv_test, y_test, best_params)

    report_table = tabulate([
        ['AUC', fsg_od_metrics['auc']],
        ['AUC Ratio', fsg_od_metrics['auc_ratio']],
        ['AP Ratio', fsg_od_metrics['ap_ratio']],
        ['Precision Ratio', fsg_od_metrics['precision_ratio']],
        ['Fairness', fsg_od_metrics['fairness']],
        ['Group Fidelity', fsg_od_metrics['group_fidelity']],
    ], headers=['SG-FOD'])

    print("Metrics report...")
    print(report_table)
    print()

    optimization_report = {
        'test_metrics': fsg_od_metrics,
        'optim_best_score': best_score_train,
        'optim_best_params': best_params
    }

    with open(report_path + f"/optim_{dataname}.json", 'w') as file:
        json.dump(optimization_report, file, indent=4)


def opt_insurance(seed=33, 
              datasets_base='datasets',
              bayesian_runs=32,
              report_path='outputs/optimizations'):

    dataname = "insurance"

    print("Running Bayesian optimization for the KDD dataset...")
    tf.random.set_seed(seed)
    np.random.seed(seed)
    
    insurance_df = pd.read_csv(f"{datasets_base}/proc/crafted_insurance.csv")

    y_pv = insurance_df[['target','sex']]
    X = insurance_df.drop(columns=['target', 'sex'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True, random_state=seed)

    pv_test = y_pv_test['sex']
    pv_train = y_pv_train['sex']
    y_train = y_pv_train['target']
    y_test = y_pv_test['target']

    best_params, best_score_train = run_optim(X_train, y_train, pv_train, bayesian_runs=bayesian_runs, seed=seed)

    print("Optimization done")

    print(
        tabulate(
            pd.DataFrame.from_dict(best_params, orient='index')
        )
    )

    print(f"score:  {best_score_train.astype(float)}")
    print()

    fsg_od_metrics = run_eval(X_train, pv_train, X_test, pv_test, y_test, best_params)

    report_table = tabulate([
        ['AUC', fsg_od_metrics['auc']],
        ['AUC Ratio', fsg_od_metrics['auc_ratio']],
        ['AP Ratio', fsg_od_metrics['ap_ratio']],
        ['Precision Ratio', fsg_od_metrics['precision_ratio']],
        ['Fairness', fsg_od_metrics['fairness']],
        ['Group Fidelity', fsg_od_metrics['group_fidelity']],
    ], headers=['SG-FOD'])

    print("Metrics report...")
    print(report_table)
    print()

    optimization_report = {
        'test_metrics': fsg_od_metrics,
        'optim_best_score': best_score_train,
        'optim_best_params': best_params
    }

    with open(report_path + f"/optim_{dataname}.json", 'w') as file:
        json.dump(optimization_report, file, indent=4)



def main(dataset, bayesian_runs):
    avail_datasets = {
        'adult': opt_adult,
        'bank': opt_bank,
        'credit': opt_credit,
        'kdd': opt_kdd,
        'obesity': opt_obesity,
        'insurance': opt_insurance
    }

    if dataset == 'all':
        for _, item in avail_datasets.items():
            item(bayesian_runs=bayesian_runs)
    else:
        avail_datasets[dataset](bayesian_runs=bayesian_runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparam opt script.')
    parser.add_argument('dataset', metavar='dataset', type=str, nargs=1,
                        help='the dataset you want to run bayesian opt on. "all" for building all of them',
                        choices=['all', 'adult', 'bank', 'credit', 'insurance', 'kdd', 'obesity'])
    parser.add_argument('--br', metavar='--bayesian_runs', type=int, nargs=1,
                        help='Numer of runs of the bayesian optimization algorithm',
                        default=32

    )
    

    args = parser.parse_args()

    dataset = args.dataset[0]
    bayesian_runs = args.br[0]

    main(dataset, bayesian_runs)