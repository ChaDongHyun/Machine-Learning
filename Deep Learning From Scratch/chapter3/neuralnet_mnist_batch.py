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


batch_size = 100  # 배치 크기
accuracy_cnt = 0


for i in range(0, len(x), batch_size):
    # range()함수는 range(start, end, step)으로 지정하면 step 간격으로 증가하는 리스트 반환
   x_batch = x[i:i+batch_size]  # 배치사이즈인 100개씩 슬라이싱
   y_batch = predict(network, x_batch)
   p = np.argmax(y_batch, axis=1)  # 100*10의 배열 중에서 1차원을 구성하는 각 원소에서 최댓값의 인덱스 찾음
   accuracy_cnt += np.sum(p == t[i:i+batch_size])  #분류한 결과를 실제 답과 비교

print("Accuracy:" + str(float(accuracy_cnt)/ len(x)))