from sklearn import metrics
import numpy as np
from metrics.affiliation.generics import convert_vector_to_events

from metrics.affiliation.metrics import pr_from_events


def auc_roc(y_true, y_score):

    return metrics.roc_auc_score(y_true, y_score)


def auc_pr(y_true, y_score):

    
    return metrics.average_precision_score(y_true, y_score)


def tabular_metrics(y_true, y_score):

    ratio = 100.0 * len(np.where(y_true == 0)[0]) / len(y_true)
    thresh = np.percentile(y_score, ratio)
    y_pred = (y_score >= thresh).astype(int)
    y_true = y_true.astype(int)
    p, r, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred, average='binary')

    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), f1


def ts_metrics(y_true, y_score):

    best_f1, best_p, best_r, thresh = get_best_f1(y_true, y_score)

    return auc_roc(y_true, y_score), auc_pr(y_true, y_score), best_f1, best_p, best_r, thresh


def get_best_f1(label, score):
    
    precision, recall, thresh = metrics.precision_recall_curve(y_true=label, probas_pred=score)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    best_f1 = f1[np.argmax(f1)]
    best_p = precision[np.argmax(f1)]
    best_r = recall[np.argmax(f1)]
    max_thresh = thresh[np.argmax(f1)]
    return best_f1, best_p, best_r, max_thresh


def ts_metrics_enhanced(y_true, y_score, y_test):
   
    best_f1, best_p, best_r, thresh = get_best_f1(y_true, y_score)

   
    events_pred = convert_vector_to_events(y_test) 
    events_gt = convert_vector_to_events(y_true)
    Trange = (0, len(y_test))
    affiliation = pr_from_events(events_pred, events_gt, Trange)
 

    auroc = auc_roc(y_true, y_score)
    aupr = auc_pr(y_true, y_score)

    affiliation_precision = affiliation['Affiliation_Precision']
    affiliation_recall = affiliation['Affiliation_Recall']



    affiliation_f1 = 2 * affiliation_precision * affiliation_recall /(affiliation_precision + affiliation_recall)
    return {'pc_adjust': best_p,
            'rc_adjust': best_r,
            'f1_adjust': best_f1,
            'af_pc': affiliation_precision,
            'af_rc': affiliation_recall,
            'af_f1': affiliation_f1,
         
            'auc_pr': aupr,
            'auc_roc': auroc,
            'trt': 0.0,
            'tst': 0.0
            }
