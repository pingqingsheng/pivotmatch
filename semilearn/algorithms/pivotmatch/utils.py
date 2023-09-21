# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook

def random_inital(x_feature, x_target, x_conf, num_pivots=200, num_classes=10, threshold=0.95):
    k_per_class = num_pivots // num_classes
    q_initial = []
    q_label = []
    idx_initial = []

    for class_id in range(num_classes):
        idx_class = torch.where(torch.logical_and(x_target == class_id, x_conf > threshold))[0]
        idx_selected = idx_class[np.random.choice(np.arange(idx_class.shape[0]), k_per_class)]

        idx_initial.append(idx_selected)
        feature_selected = x_feature[idx_selected]
        q_initial.append(feature_selected)
        q_label.append([class_id] * k_per_class)

    q_initial = torch.cat(q_initial, 0)
    q_label = torch.tensor(q_label).long().flatten()
    # idx_initial = torch.stack(idx_initial)
    # idx_initial_np = idx_initial.reshape(-1).cpu().numpy()

    # return q_initial, q_label, idx_initial, idx_initial_np
    return q_initial.to(x_feature.device), q_label.to(x_feature.device)


def topk_inital(x_feature, x_target, x_conf, num_pivots=200, num_classes=10):
    k_per_class = num_pivots // num_classes
    q_initial = []
    q_label = []
    idx_initial = []

    for class_id in range(num_classes):
        idx_class = torch.where(x_target == class_id)[0]
        conf_class = x_conf[idx_class]
        idx_selected = idx_class[torch.argsort(conf_class, descending=True)[:k_per_class]]

        # idx_initial.append(idx_selected)
        feature_selected = x_feature[idx_selected]
        q_initial.append(feature_selected)
        q_label.append([class_id] * k_per_class)

    q_initial = torch.cat(q_initial, 0)
    q_label = torch.tensor(q_label).long().flatten()
    # idx_initial = torch.stack(idx_initial)
    # idx_initial_np = idx_initial.reshape(-1).cpu().numpy()

    # return q_initial, q_label, idx_initial, idx_initial_np
    return q_initial.to(x_feature.device), q_label.to(x_feature.device)


@torch.no_grad()
def ot_sinkhorn(M, r=None, c=None, epsilon=0.05, err=1e-8, max_iters=3, exp=True, multi_gpu=False):
    """
    Computes the optimal transport matrix and Slinkhorn distance using the
    Sinkhorn-Knopp algorithm
    Inputs:
        - M : cost matrix (n x m) 
              M = -A in pivot based clustering 
              M = -log(P) in distribution alignment of cluster label  
        - r : vector of marginals (m, ) of row
        - c : vector of marginals (n, ) of column
        - epsilon : parameter of the entropic regularization
        - err : convergence parameter
    Outputs:
        - cluster_assignment : cluster assginment matrix (n x m)
    """

    err = torch.tensor(err).to(M.device)
    n, m = M.shape

    # update cost matrix
    if exp:
        Q = torch.exp(- M / epsilon)
    else:
        # M = -logP
        Q = (-M) ** (1 / epsilon) 

    # normalize Q
    sum_Q = Q.sum()    
    if multi_gpu:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    # uniform distribution by default
    if r is None or c is None:
        r = torch.ones((n)).to(M.device) / n
        c = torch.ones((m)).to(M.device) / m

    v = torch.zeros(m).to(M.device)
    i = 0
    # normalize this matrix
    while torch.max(torch.abs(v -  Q.sum(0))) > err:
        v = Q.sum(0)
        if multi_gpu:
            torch.distributed.all_reduce(v)
        Q *= (c / v).reshape((1, -1)) 
        u = Q.sum(1)
        Q *= (r / u).reshape((-1, 1))
        i += 1
        if i >= max_iters:
            break
    # cluster_assignment = Q / r.reshape(-1, 1)
    cluster_assignment = Q / Q.sum(1, keepdim=True) 
    # w_distance = torch.sum(Q * M)
    # return Q, cluster_assignment, i
    return cluster_assignment
