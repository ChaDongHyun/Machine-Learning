import sys, os
import numpy as np
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))  # numpy로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()


# 처음 한번은 몇 분 정도 걸림
(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)  # flatten = True 면 1차원 numpy 배열로 저장됨
#  normalize는 입력 이미지의 픽셀을 0.0~1.0 사이의 값으로 정규화할지 결정
#  False 이면 입력 이미지의 픽셀은 원래값 0~255 사이의 값을 유지

# 각 데이터의 형상 출력
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)  # (10000, 784)
print(t_test.shape)  # (10000,)

img = x_train[0]
label = t_train[0]
print(label)  # 5

print(img.shape)
img = img.reshape(28, 28)  # 원래 형상인 28 * 28 크기로 다시 변형
print(img.shape)


img_show(img)