
from metrics.links.generics import convert_vector_to_events
from metrics.links.metrics import pr_from_events

import numpy as np

def combine_all_evaluation_scores(y_test, pred_labels):
    events_pred = convert_vector_to_events(y_test) 
    events_gt = convert_vector_to_events(pred_labels)
    Trange = (0, len(y_test))
    links = pr_from_events(events_pred, events_gt, Trange)
    aff_p, aff_r = links['Links_Precision'], links['Links_Recall']
    aff_f1 = 2 * (aff_p * aff_r) / (aff_p + aff_r)
    pa_accuracy, pa_precision, pa_recall, pa_f_score = get_adjust_F1PA(y_test, pred_labels)

    score_list_simple = {
                  "Links precision": aff_p,
                  "Links recall": aff_r,
                  "Links f1 score": aff_f1,

                  }

    return score_list_simple


def get_adjust_F1PA(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import accuracy_score

    accuracy = accuracy_score(gt, pred)
    precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                          average='binary')
    return accuracy, precision, recall, f_score

