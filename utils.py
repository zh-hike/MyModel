from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import f1_score
from munkres import Munkres
import numpy as np


def eva(y_true, y_pred, n_clusters):
    """
    计算评估结果，acc,nmi,ari,f1
    """
    y_true = y_true - y_true.min()

    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)

    l1 = list(set(y_true))
    l2 = list(set(y_pred))

    cost = np.zeros((n_clusters, n_clusters), dtype=int)

    for i in l1:

        indexs = [i1 for i1 in range(len(y_true)) if y_true[i1] == i]  # 记录i类别的索引
        for j in l2:
            c = [j1 for j1 in indexs if y_pred[j1] == j]
            cost[(i, j)] = len(c)

    m = Munkres()
    cost = -cost
    indexs = m.compute(cost)  # 记录最佳match
    new_x = np.zeros_like(y_pred)
    for i in indexs:
        end = i[1]
        y_pred_index = [i1 for i1 in range(len(y_pred)) if y_pred[i1] == end]
        new_x[y_pred_index] = i[0]

    acc = acc_score(y_true, new_x)
    f1 = f1_score(y_true, new_x.tolist(), average='macro')

    return acc, nmi_score, ari, f1
