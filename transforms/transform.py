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

def add_ampl_diff(data):
    data['ampl_diff'] = np.concatenate([np.array([[0.0]]), data['ampl'][1:] - data['ampl'][:-1]])
    return data

# Run before norm_time
def get_angle(data):
    data['time'] = data['time'] / TIME_NORMALIZER
    p1_x = data['time'][:-2]
    p2_x = data['time'][1:-1]
    p3_x = data['time'][2:]

    p1_y = data['ampl'][:-2]
    p2_y = data['ampl'][1:-1]
    p3_y = data['ampl'][2:]
    result = np.arctan2(p3_y - p1_y, p3_x - p1_x) - \
                np.arctan2(p2_y - p1_y, p2_x - p1_x)
    result = np.concatenate([np.array([[0.0]]), result, np.array([[0.0]])], axis=0)
    data['angle'] = result
    return data

def apply_iteratively(functions):
    def _inner_func(data):
        for func in functions:
            data = func(data)
        return data

    return _inner_func

def get_test_transform(opt):
    return apply_iteratively([norm, get_angle, norm_time, add_ampl_diff])


def get_train_transform(opt):
    return apply_iteratively([norm, get_angle, norm_time, add_ampl_diff])