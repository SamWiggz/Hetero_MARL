import colorsys

import numpy as np


def positive_int(name, value):
    value = int(value)
    if value < 1:
        raise ValueError("{} must be at least 1.".format(name))
    return value


def nonnegative_int(name, value):
    value = int(value)
    if value < 0:
        raise ValueError("{} cannot be negative.".format(name))
    return value


def indexed_color(index, total):
    total = max(1, int(total))
    hue = float(index % total) / total
    return np.array(colorsys.hsv_to_rgb(hue, 0.75, 0.85))
