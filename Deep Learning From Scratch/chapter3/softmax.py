import numpy as np

'''
# exp의 값이 커지면 오버플로우 발생
def softmax(a):
    exp_a = np.exp(a)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp

    return y
'''

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #오버플로우 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)

print(np.sum(y))