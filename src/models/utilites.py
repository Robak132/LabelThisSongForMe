import torch
from sklearn import metrics


def get_auc(est_array, gt_array):
    roc_aucs = metrics.roc_auc_score(gt_array, est_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    print('roc_auc: %.4f' % roc_aucs)
    print('pr_auc: %.4f' % pr_aucs)
    return roc_aucs, pr_aucs


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x
