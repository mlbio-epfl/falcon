import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
def cluster_acc(y_pred, y_true):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    assert y_pred.shape == y_true.shape
    w = torch.zeros(y_pred.max()+1, y_true.max()+1).long()
    for i in range(y_pred.size(0)):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    mapping = torch.ones(y_pred.max() + 1).long() * -1
    mapping[row_ind] = torch.from_numpy(col_ind)
    accuracy = (w[row_ind, col_ind].sum() / y_true.size(0)).item() * 100
    return accuracy

def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean().item() * 100


def macro_accuracy(y_pred, y_true, return_class_accuracies=False):
    unique_classes = np.unique(y_true)

    class_accuracies = []
    for cls in unique_classes:
        class_indices = np.where(y_true == cls)
        class_true = y_true[class_indices]
        class_pred = y_pred[class_indices]

        class_accuracy = accuracy(class_pred, class_true)
        class_accuracies.append(class_accuracy)
    macro_accuracy = np.mean(class_accuracies)
    if return_class_accuracies:
        return macro_accuracy, class_accuracies
    return macro_accuracy

def compute_matchings(y_pred, y_true):
    assert y_pred.shape == y_true.shape
    D = max(y_pred.max(), y_true.max()) + 1
    w = torch.zeros(D, D).long()
    for i in range(y_pred.size(0)):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    mapping = torch.ones(D).long() * -1000
    mapping[row_ind] = torch.from_numpy(col_ind)
    return mapping

def normalized_mutual_info_score(y_pred, y_true):
    from sklearn.metrics import normalized_mutual_info_score
    return normalized_mutual_info_score(y_pred, y_true) * 100.

def adjusted_rand_score(y_true, y_pred):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(y_true, y_pred) * 100.

def cm_cluster_acc(cm):
    row_ind, col_ind = linear_sum_assignment((cm.max() - cm).cpu().numpy())
    accuracy = (cm[row_ind, col_ind].sum() / cm.sum()).item() * 100
    return accuracy
