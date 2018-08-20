import numpy as np
import matplotlib.pylab as plt

''''
def step_function(x): #numpy 입력으로 못받음
    if x > 0:
        return 1
    else:
        return 0
'''''

## 입력값을 부등호로 받으면 부울값으로 변환
## np.int형으로 변환

def step_function(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1) # -5.0에서 5.0 전까지 0.1 간격의 넘파이 배열 생성 [-5.0, -4.9, ..., 4.9]
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 범위 지정
plt.show()
