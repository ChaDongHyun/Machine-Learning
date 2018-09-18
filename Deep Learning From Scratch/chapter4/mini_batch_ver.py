import numpy as np
import sys, os
from dataset.mnist import load_mnist
sys.path.append(os.pardir)


def cross_entropy_error(y, t):  # mini batch 전용 교차 엔트로피 오차
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def non_one_hot_cee(y, t):  # mini batch 전용 교차 엔트로피 (정답 레이블이 one-hot 인코딩 아닌 경우)
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t]) + 1e-7) / batch_size
    # np.arange(batch_size)를 하면 0부터 batch_size -1 까지의 넘파이 배열 생성


(x_train, t_train), (x_test, t_test) =  \
    load_mnist(normalize=True, one_hot_label=True)

# 훈련데이터 60,000개, 입력데이터 784열, 정답레이블 10줄
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10  # 배치사이즈 10으로
batch_mask = np.random.choice(train_size, batch_size)  # 60000개의 훈련데이터중 10개(batch_size) 무작위 선택
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]