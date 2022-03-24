from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch.nn.functional as F


def Standard(data, standard):
    # print(standard)
    if standard is None:
        return data
    if standard == 'L2':
        return F.normalize(data, p=2)

    method = {'MinMax': MinMaxScaler([-1, 1]),
              'Standard': StandardScaler(),
              }
    preprocess = method[standard]

    data = preprocess.fit_transform(data)

    return data.astype('float32')
