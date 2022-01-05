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
from sklearn.metrics import roc_auc_score

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# SEED = 33
# tf.random.set_seed(SEED)
# np.random.seed(SEED)


def run_experiment(
    dataset_path='datasets/proc/crafted_adult.csv',
    report_path='outputs/report_adult.json'
):
    # Here we'll save the metrics report
    report_metrics = dict()

    # Load adult dataset
    adult_df = pd.read_csv(dataset_path)
    #adult_df = adult_df.loc[np.zeros(len(adult_df), dtype=int), :]

    y_pv = adult_df[['OUTLIER','sex']]
    X = adult_df.drop(columns=['OUTLIER', 'sex'])

    X_train, X_test, y_pv_train, y_pv_test = train_test_split(X, y_pv, test_size=0.25, shuffle=True, random_state=42)

    pv_test = y_pv_test['sex']
    pv_train = y_pv_train['sex']
    y_test = y_pv_test['OUTLIER']


    print("Training base fSG-OD...")
    model = SGOutlierDetector(
                          epsilon_p=90, 
                          lambda_se=0.01, 
                          lambda_a=20,
                          a=6, 
                          embedding_dim=5,
                          alpha=0.4, 
                          gamma=0.25)

    _, _ = model.fit(X_train, pv_train, batch_size=1024, epochs=100, verbose=True, early_stop=True)

    print("Evaluating fSG-OD...")
    X_pred = model.predict_scores(X_test).numpy()

    #fsg_od_metrics = {
    #    'auc': roc_auc_score(y_test, X_pred).astype(float),
    #    'auc_ratio': evaluation.auc_ratio(y_test, X_pred, pv_test).astype(float),
    #    'ap_ratio': evaluation.compute_AP_ratio(y_test, X_pred, pv_test).astype(float),
    #    'precision_ratio': evaluation.compute_precision_ratio(y_test, X_pred, pv_test).astype(float),
    #    'fairness': evaluation.compute_Fairness_metric(y_test, X_pred, pv_test).numpy().astype(float),
    #    'group_fidelity': evaluation.compute_GF_metric(X_pred, pv_test).numpy().astype(float)
    #}
    #report_metrics['metrics_fsgod'] = fsg_od_metrics
#
    #report_table = tabulate([
    #    ['AUC', report_metrics['metrics_fsgod']['auc'], report_metrics['metrics_fsgod']['auc']],
    #    ['AUC Ratio', report_metrics['metrics_fsgod']['auc_ratio'], report_metrics['metrics_fsgod']['auc_ratio']],
    #    ['AP Ratio', report_metrics['metrics_fsgod']['ap_ratio'], report_metrics['metrics_fsgod']['ap_ratio']],
    #    ['Precision Ratio', report_metrics['metrics_fsgod']['precision_ratio'], report_metrics['metrics_fsgod']['precision_ratio']],
    #    ['Fairness', report_metrics['metrics_fsgod']['fairness'], report_metrics['metrics_fsgod']['fairness']],
    #    ['Group Fidelity', report_metrics['metrics_fsgod']['group_fidelity'], report_metrics['metrics_fsgod']['group_fidelity']]
    #], headers=['SG-FOD', 'SG-FOD'])

    #

    print("Metrics report...")
    #print(report_table)
    print()
    #import pdb; pdb.set_trace()
    print(roc_auc_score(y_test, X_pred).astype(float))

    # if roc_auc_score(y_test, X_pred).astype(float) < 0.6:
    #     import pdb; pdb.set_trace()

def main():
    run_experiment()


if __name__ == "__main__":
    main()