import numpy as np
import matplotlib.pylab as plt

"""
# 나쁜 구현의 예
def numerical_diff(f, x):
    h = 10e-50
    return (f(x+h) - f(x)) / h

# h에 너무 작은 값이 들어가 반올림 오차(rounding error)가 발생하여 최종 계산 결과에 오차 발생
# f(x+h)와 f(x)의 차분에서 오차가 존재, f(x+h)와 f(x) 사이의 기울기가 구해짐.
# 구하려는 기울기는 f(x)의 기울기이고, 이게 진정한 미분
"""


def numerical_diff(f, x):
    h = 1e-4  # 0.0001 # h의 미세한 값으로 10^-4을 사용하는것이 좋다고 알려져 있다.
    return (f(x+h) - f(x-h)) / (2*h)
    # x를 중심으로 h만큼의 전후의 차분을 계산하는 중심 차분, 분모는 폭이 2h 이기 때문.


def function_1(x):
    return 0.01 * x**2 + 0.1 * x  # 0.01x^2 + 0.1x


x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()

print(numerical_diff(function_1, 5))  # x가 5 일때의 미분
print(numerical_diff(function_1, 10))  # x가 10 일때의 미분


def function_2(x):  # f(x0, x1) = x0^2 + x1^2
    return np.sum(x**2)  # return x[0]**2 + x[1]**2로 표현 할 수도 있음


def function_tmp1(x0):  # x0 = 3, x1 = 4 일 때, x0에 대한 편미분 구하기 위한 함수식
    return x0*x0 + 4.0**2.0


def function_tmp2(x1):  # x0 = 3, x1 = 4 일 때, x1에 대한 편미분 구하기 위한 함수식
    return 3.0**2.0 + x1*x1


print(numerical_diff(function_tmp1, 3.0))  # x0 = 3, x1 = 4 일 때, x0에 대한 편미분
print(numerical_diff(function_tmp2, 4.0))  # x0 = 3, x1 = 4 일 때, x1에 대한 편미분