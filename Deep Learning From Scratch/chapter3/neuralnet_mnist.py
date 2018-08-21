import numpy as np
import os, sys
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
sys.path.append(os.pardir)


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test
# normalize=True 이므로 정규화로 전처리(단순히 255로 나누어 0.0~1.0 범위의 값으로 변환)


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']


    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()


accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])  # predict 함수는 각 레이블의 확률을 넘파이 배열로 반환
    p = np.argmax(y)  # 확률이 제일 높은 원소의 인덱스 반환
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt)/ len(x)))