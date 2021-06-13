import numpy as np

MEDIAN = 624.0
O_25 = 548.0
O_75 = 732.0

TIME_NORMALIZER = 624.0

def norm(data):
    data['ampl'] = (data['ampl'] - MEDIAN) / (O_75 - O_25)
    return data

def norm_time(data):
    data['time'] = np.concatenate([np.array([[0.0]]), data['time'][1:] - data['time'][:-1]])
    data['time'] = data['time'] / TIME_NORMALIZER
    return data


def apply_iteratively(functions):
    def _inner_func(data):
        for func in functions:
            data = func(data)
        return data

    return _inner_func

def get_test_transform(opt):
    return apply_iteratively([norm, norm_time])


def get_train_transform(opt):
    return apply_iteratively([norm, norm_time])