import numpy as np


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))  # delta 라는 아주 작은 값을 더해 y가 0일때 값이 무한대가 되는걸 막음