import os
import sys
import inspect

import numpy as np
import tensorflow as tf
import pandas as pd
import json

from tabulate import tabulate

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from fairsgod.losses import FairODLoss
from fairsgod.fairod import OutlierDetector
from fairsgod.fairod_sg import SGOutlierDetector
from fairsgod import evaluation

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# SEED = 33
# tf.random.set_seed(SEED)
# np.random.seed(SEED)


def run_experiment(
    dataset_path='datasets/proc/crafted_celeba.csv',
    report_path='outputs/report_celeba.json'
):
    # Here we'll save the metrics report
    report_metrics = dict()

    # Load celebA dataset
    celeba_df = pd.read_csv(dataset_path)
    celeba_df = celeba_df.astype(np.float64)
    y_pv = celeba_df[['Male','Attractive']]
    X = celeba_df.drop(columns=['Male','Attractive'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.3, shuffle=True)

    pv_test = y_pv_test['Male']
    pv_train = y_pv_train['Male']
    y_test = y_pv_test['Attractive']


    # Training FairOD Model
    print("Training base FairOD...")
    model = OutlierDetector(alpha=0.4, gamma=0.25)
    _, _ = model.fit(X_train, pv_train, batch_size=512, epochs=6, val_X=None, val_pv=None)

    print("Evaluating FairOD...")
    X_pred = model.predict_scores(X_test).numpy()
    X_pred = np.nan_to_num(X_pred)  # In case FairOD predicts NaNs


    fair_od_metrics = {
        'auc': roc_auc_score(y_test, X_pred).astype(float),
        'auc_ratio': evaluation.auc_ratio(y_test, X_pred, pv_test).astype(float),
        'ap': average_precision_score(y_test, X_pred).astype(float),
        'ap_ratio': evaluation.compute_AP_ratio(y_test, X_pred, pv_test).astype(float),
        'precision_ratio': evaluation.compute_precision_ratio(y_test, X_pred, pv_test).astype(float),
        'fairness': evaluation.compute_Fairness_metric(y_test, X_pred, pv_test).numpy().astype(float),
        'group_fidelity': evaluation.compute_GF_metric(X_pred, pv_test).numpy().astype(float)
    }

    report_metrics['metrics_fairod'] = fair_od_metrics

    print("Training base fSG-OD...")
    model = SGOutlierDetector(
                          epsilon_p=90, 
                          lambda_se=0.01, 
                          lambda_a=20,
                          a=6, 
                          alpha=0.4, 
                          gamma=0.25)

    _, _ = model.fit(X_train, pv_train, batch_size=1024, epochs=40, val_X=None, val_pv=None, early_stop=True)

    print("Evaluating fSG-OD...")
    X_pred = model.predict_scores(X_test).numpy()

    fsg_od_metrics = {
        'auc': roc_auc_score(y_test, X_pred).astype(float),
        'auc_ratio': evaluation.auc_ratio(y_test, X_pred, pv_test).astype(float),
        'ap': average_precision_score(y_test, X_pred).astype(float),
        'ap_ratio': evaluation.compute_AP_ratio(y_test, X_pred, pv_test).astype(float),
        'precision_ratio': evaluation.compute_precision_ratio(y_test, X_pred, pv_test).astype(float),
        'fairness': evaluation.compute_Fairness_metric(y_test, X_pred, pv_test).numpy().astype(float),
        'group_fidelity': evaluation.compute_GF_metric(X_pred, pv_test).numpy().astype(float)
    }
    report_metrics['metrics_fsgod'] = fsg_od_metrics

    report_table = tabulate([
        ['AUC', report_metrics['metrics_fairod']['auc'], report_metrics['metrics_fsgod']['auc']],
        ['AUC Ratio', report_metrics['metrics_fairod']['auc_ratio'], report_metrics['metrics_fsgod']['auc_ratio']],
        ['AP', report_metrics['metrics_fairod']['ap'], report_metrics['metrics_fsgod']['ap']],
        ['AP Ratio', report_metrics['metrics_fairod']['ap_ratio'], report_metrics['metrics_fsgod']['ap_ratio']],
        ['Precision Ratio', report_metrics['metrics_fairod']['precision_ratio'], report_metrics['metrics_fsgod']['precision_ratio']],
        ['Fairness', report_metrics['metrics_fairod']['fairness'], report_metrics['metrics_fsgod']['fairness']],
        ['Group Fidelity', report_metrics['metrics_fairod']['group_fidelity'], report_metrics['metrics_fsgod']['group_fidelity']]
    ], headers=['FairOD', 'SG-FOD'])

    #

    print("Metrics report...")
    print(report_table)
    print()
    print(f"Saving report at: {report_path}")

    #import pdb; pdb.set_trace()

    with open(report_path, 'w') as file:
        json.dump(report_metrics, file, indent=4)


def main():
    run_experiment()


if __name__ == "__main__":
    main()