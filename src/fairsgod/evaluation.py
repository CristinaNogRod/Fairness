import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score, roc_curve, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
from tensorflow.python.training.tracking import base
import numpy as np

def compute_threshold(y_true, outlier_score):
    """
    This function finds the outlier score threshold. 
    It counts the number of true outliers from true label (say x), then sorts the scores and returns the xth score.

    Args:
        y_true: true labels (outlier, inlier) 
        outlier_scores: mse of sample

    Returns:
        double: the threshold
    """
    _, _, counts = tf.unique_with_counts(y_true)
    n_outliers = counts[1].numpy()
    sorted_outliers = tf.sort(outlier_score, direction='DESCENDING')
    return sorted_outliers[n_outliers-1]


def outlier_classifier(threshold, outlier_scores):
    """
    This function flags samples as outliers or not.

    Args:
        threshold: if outlier_score is larger than this value, then the sample is classified as an outlier
        outlier_scores: tensor of MSEs

    Returns:
        binary tensor: 1 - outlier, 0 - inlier
    """
    r = outlier_scores >= threshold
    return  tf.cast(r,tf.int64)



def compute_outlier_scores(y_true, y_pred):
    """
    This function computes the MSE of the autoencoder output.

    Args:
        y_true: tensor of inputs passed through autoencoder 
        y_pred: tensor of outputs 

    Returns:
        for each sample, returns the MSE 
    """
    return tf.reduce_mean((y_true-y_pred)**2, axis=1)



def eval(outlier_scores, y_true, threshold):
    """
    This function computes evaluation metrics:
        - f1_score, precision, recall for outliers and inliers
        - auc_score
        - roc curve
        - confusion matrix
    
    Args:
        outlier_scores: tensor of MSEs
        y_true: tensor of target labels
        threshold: used to classify outlier_scores into outlier or inlier group

    Returns:
        confusion matrix (and prints other metrics/plots)
    """
    y_pred = outlier_classifier(threshold, outlier_scores)

    print("\n AUC score: ", roc_auc_score(y_true, outlier_scores))
    report = classification_report(y_true, y_pred, output_dict=True)
    print("\n", pd.DataFrame(report).transpose().iloc[0:2, 0:3], "\n")

    fpr , tpr , _ = roc_curve(y_true, outlier_scores)
    plot_roc_curve(fpr,tpr)
    fig, ax = plt.subplots(1, 1)
    conf = confusion_matrix(y_true, y_pred, normalize='all')
    sns.heatmap(conf, annot=True, ax=ax)
    plt.show()
    return 



def plot_roc_curve(fpr,tpr): 
    """ 
    This function generates the roc curve.
    """
    fig = plt.figure()
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    return 




def compute_AP_ratio(y_true, outlier_scores, pv):
    """
    This function computes the average precision ratio between the protected groups.

    Args:
        y_true: tensor of true labels
        outlier_score: MSEs of samples
        pv: tensor of protected variable labels (binary: 0-majority, 1_minority)
    
    Returns:
        ratio = AP(majority group) / AP(minority group)
    """
    y_true_maj = y_true[pv==0]
    y_true_min = y_true[pv==1]
    y_pred_maj = tf.boolean_mask(outlier_scores, pv==0)
    y_pred_min = tf.boolean_mask(outlier_scores, pv==1)

    return average_precision_score(y_true_maj, y_pred_maj) / average_precision_score(y_true_min, y_pred_min)


def compute_precision_ratio(y_true, outlier_scores, pv, threshold=None):
    """ 
    This function computes the precision ratio between protected groups.

    Args: 
        y_true: tensor of true labels
        outlier_score: MSEs of samples
        pv: tensor of protected variable labels (binary: 0-majority, 1_minority)

    Returns:
        ratio = precision(majority group) / precision(minority group)
    """
    y_true_maj = y_true[pv==0]
    y_true_min = y_true[pv==1]

    if not threshold:
        threshold = compute_threshold(y_true, outlier_scores)
    y_pred = outlier_classifier(threshold, outlier_scores)
    y_pred_maj = tf.boolean_mask(y_pred, pv==0)
    y_pred_min = tf.boolean_mask(y_pred, pv==1)

    return precision_score(y_true_maj, y_pred_maj) / precision_score(y_true_min, y_pred_min)


def compute_Fairness_metric(y_true, outlier_scores, pv, threshold=None):
    """
    This function measures fairness in terms of statistical parity.
    It computes the flag-rate ratio per protected group.

    Args:
        outlier_scores: MSEs of samples
        threshold: float value used to classify samples as either outlier or not
        pv: tensor of protected labels

    Returns:
        minimum of flag-rate ratio and 1/flag-rate
    """
    if not threshold:
        threshold = compute_threshold(y_true, outlier_scores)
    y_pred = outlier_classifier(threshold, outlier_scores)

    y_pred_maj = tf.boolean_mask(y_pred, pv==0)
    y_pred_min = tf.boolean_mask(y_pred, pv==1)

    flag_ratio_maj = tf.reduce_sum(y_pred_maj) / len(y_pred_maj)
    flag_ratio_min = tf.reduce_sum(y_pred_min) / len(y_pred_min)

    r = flag_ratio_maj / flag_ratio_min

    return min(r, 1/r)



def compute_GF_metric(outlier_scores, pv):
    """
    This function measure how well the group ranking of base detector is preserved in the fair-aware detectors.
    
    Args:
        outlier_scores: MSEs of samples
        pv: tensor of protected labels

    Returns:
        harmonic mean (per protected group) of NDCG
    """
    outlier_scores_maj = tf.boolean_mask(outlier_scores, pv==0)
    outlier_scores_min =tf.boolean_mask(outlier_scores, pv==1)

    frac_maj = 1/NDCG(outlier_scores_maj)
    frac_min = 1/NDCG(outlier_scores_min)

    return 2/(frac_min + frac_maj) 


def NDCG(outlier_scores):
    """
    This function computes the NDCG (Normalised Discounted Cumulative Gain).
    """
    indicator = tf.math.less_equal(outlier_scores, tf.expand_dims(outlier_scores, axis=1)) 
    denom_sum = tf.math.reduce_sum(  tf.cast(indicator , tf.float32), axis=0 )
    log_term = tf.experimental.numpy.log2(1. + denom_sum)
    return tf.math.reduce_sum( tf.cast((2**outlier_scores - 1), tf.float32) / (log_term * IDCG(outlier_scores)) )


def IDCG(outlier_scores):
    """
    This function computes the IDCG (Ideal Discounted Cumulative Gain) used to normalise DCG.
    """
    j = tf.linspace(1., len(outlier_scores), len(outlier_scores))
    return tf.math.reduce_sum( tf.cast((2.**outlier_scores - 1.), tf.float32) /  tf.experimental.numpy.log2(1. + j) )




def compute_rankAgreement_metric(y_true, os_base, os_fair):
    """
    This function measures how well the final ranking of the FairOD aligns with the ranking of the baseOD.
    Given two tensors, it sorts them in descending order, then selects the top_k per tensor
        and computes the number of samples in the intersection divided by those in the union.

    Args:
        os_base: outlier scores (mse) of base detector
        os_fair: outlier scores of fair detector for the same sample as os_base
        n_outiers: number of true outliers (in unsupervised setting this is the users choice)
    
    Returns:
        number of samples in intersection/ number of samples in union
    """
    _, _, counts = tf.unique_with_counts(y_true)
    n_outliers = counts[1].numpy()
    base_sorted = tf.argsort(os_base, direction='DESCENDING').numpy()
    fair_sorted = tf.argsort(os_fair, direction='DESCENDING').numpy()
    base_sorted = base_sorted[:n_outliers]
    fair_sorted = fair_sorted[:n_outliers]

    intersect = len(np.intersect1d(base_sorted, fair_sorted))
    union = len(np.union1d(base_sorted, fair_sorted))

    return intersect/union


def auc_ratio(ytrue, y_pred, pv):
    pv = pv.astype(bool).values
    auc_maj = roc_auc_score(ytrue[pv], y_pred[pv])
    auc_min = roc_auc_score(ytrue[~pv], y_pred[~pv])
    return auc_maj/auc_min