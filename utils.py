from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import accuracy_score as acc_score
from sklearn.metrics import f1_score
from munkres import Munkres
import numpy as np
import torch
import random





def eva(y_true, y_pred, n_clusters):
    """
    计算评估结果，acc,nmi,ari,f1
    """
    y_true = y_true.astype('int')
    y_pred = y_pred.astype('int')
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


def get_missmatrix(miss_rate, n_view, n_sample):
    miss_matrix = np.ones((n_sample, n_view))
    all_index = list(range(n_sample))
    miss_num = int(miss_rate * n_sample)
    miss_index = random.sample(all_index, miss_num)
    miss_num_of_a_sample = list(range(1, n_view))
    miss_index_of_a_sample = list(range(0, n_view))
    for index in miss_index:
        miss_num_this_sample = random.choice(miss_num_of_a_sample)
        m = random.sample(miss_index_of_a_sample, miss_num_this_sample)
        miss_matrix[index][m] = 0

    miss_matrix = torch.from_numpy(miss_matrix.astype('int'))
    miss_matrixs = [miss_matrix.T[i].unsqueeze(1).numpy() for i in range(n_view)]
    return miss_matrixs
