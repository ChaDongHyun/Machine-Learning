import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x와 형상이 같고 그 원소가 모두 0인 배열을 생성

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
    return np.sum(x**2)  # return x[0]**2 + x[1]**2로 표현 할 수도 있음


print(numerical_gradient(function_2, np.array([3.0, 4.0])))  # (3,4)에서의 기울기
print(numerical_gradient(function_2, np.array([0.0, 2.0])))  # (0,2)에서의 기울기
print(numerical_gradient(function_2, np.array([3.0, 0.0])))  # (3,0)에서의 기울기