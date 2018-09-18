import numpy as np


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x  # x 초기화

    for i in range(step_num):  # 갱신 반복할 횟수를 step_num으로 조절
        grad = numerical_gradient(f, x)  # numerical_gradient 함수로 기울기 구한 값을 grad에 저장
        x -= lr*grad  # x를 learning rate(학습률, 에타) * grad(기울기) 만큼 빼서 x값을 갱신
    return x


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같고 그 원소가 모두 0인 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[idx] = tmp_val -h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 값 복원
    return grad


def function_2(x):  # f(x0, x1) = x0^2 + x1^2
    return x[0] ** 2 + x[1] ** 2  # np.sum(x**2)로 표현 할 수도 있음


init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100))
print(gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100))  # 학습률이 너무 큰 예 : lr=10.0
print(gradient_descent(function_2, init_x=init_x, lr=1e-10, step_num=100))  # 학습률이 너무 작은 예 : lr=1e-10

# 학습률 같은 매개변수를 하이퍼 파라미터(hyper parameter) 라고 한다. 가중치와 편향 같은 신경망의 매개변수와는 성질이 다름.
# 신경망의 가중치 매개변수는 훈련 데이터와 학습 알고리즘에 의해서 자동으로 획득, 하이퍼파라미터는 사람이 직접 설정.
# 여러 후보 값을 시험하여 학습이 잘 되는 값을 찾는 과정 거쳐야 함
