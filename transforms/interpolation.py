import numpy as np
from scipy.interpolate import CubicSpline

CARDIO_SPIKE_MEDIAN_SAMPLE = 624

def sample_uniform(x, y, target=None, sample_rate=CARDIO_SPIKE_MEDIAN_SAMPLE):
    interpolator = CubicSpline(x, y)
    new_x = np.arange(0, max(x), sample_rate)
    new_y = interpolator(new_x)
    if target is not None:
        interpolator_target = CubicSpline(x, target)
        new_target = np.round(interpolator_target(new_x)).astype(np.int)
        return new_x, new_y, new_target
    else:
        return new_x, new_y


def sample_randomly_uniform(x, y, target=None, sample_rate=CARDIO_SPIKE_MEDIAN_SAMPLE):
    interpolator = CubicSpline(x, y)
    new_x = np.arange(0, max(x), sample_rate)
    new_x = new_x + (np.random.rand(*new_x.shape) * 2 - 1)
    new_y = interpolator(new_x)
    if target is not None:
        interpolator_target = CubicSpline(x, target)
        new_target = np.round(interpolator_target(new_x)).astype(np.int)
        return new_x, new_y, new_target
    else:
        return new_x, new_y


def sample_randomly(x, y, target=None, sample_rate=CARDIO_SPIKE_MEDIAN_SAMPLE):
    interpolator = CubicSpline(x, y)
    new_x = x + np.random.randn(*x.shape) * sample_rate
    new_y = interpolator(new_x)
    if target is not None:
        interpolator_target = CubicSpline(x, target)
        new_target = np.round(interpolator_target(new_x)).astype(np.int)
        return new_x, new_y, new_target
    else:
        return new_x, new_y


