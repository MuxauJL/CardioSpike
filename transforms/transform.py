import numpy as np
from scipy import signal

from transforms.interpolation import sample_randomly

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


def get_part_of_sequence(data, max_range=512):
    len_seq = len(data['time'])
    start_pos = np.random.randint(0, max(len_seq - max_range - 1, 1))
    data['time'] = data['time'][start_pos: start_pos + max_range]
    data['ampl'] = data['ampl'][start_pos: start_pos + max_range]
    data['target'] = data['target'][start_pos: start_pos + max_range]
    return data


def add_noise(data, std=74/2): # Half of data std
    data['ampl'] = data['ampl'] + np.random.randn(*data['ampl'].shape) * std

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

def stft(data):
    f, t, Zxx = signal.stft(data['ampl'], 1.0, nperseg=32)
    data['stft'] = np.log(np.abs(Zxx))
    return data

def random_sample(data):
    assert data['target'].shape[1] == 1
    if np.random.rand() > 0.5:
        data['time'], data['ampl'],data['target'] = sample_randomly(data['time'][:,0], data['ampl'][:,0], data['target'][:,0])
        data['time'] = np.expand_dims(data['time'], 1)
        data['ampl'] = np.expand_dims(data['ampl'], 1)
        data['target'] = np.expand_dims(data['target'], 1)
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
    if opt is not None:
        if opt.add_sampling:
            return apply_iteratively([get_part_of_sequence, random_sample,norm, get_angle, norm_time, add_ampl_diff])

    return apply_iteratively([get_part_of_sequence, norm, get_angle, norm_time, add_ampl_diff])