from sklearn.preprocessing import MinMaxScaler, StandardScaler


def Standard(data, standard):
    if standard is None:
        return data

    method = {'MinMax': MinMaxScaler([-1, 1]),
              'Standard': StandardScaler(),
              }
    preprocess = method[standard]
    data = preprocess.fit_transform(data)

    return data
