import mnist as mn
import numpy as np
from PIL import Image

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)

def getData():
    (x_train, t_train), (x_test, t_test) = mn.load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def initNetwork():
    with open("sample_weight.pkl", 'rb') as f:
        network = mn.pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    return softmax(a3)

def showImage(img):
    pil_img = Image.fromarray(np.uint8(img))  # 배열로 이미지 불러오기
    pil_img.show()

#print(x_train.shape)  # 훈련 이미지   # 60000, 784   (훈련할 이미지가 60000장이라는 소리. 가로세로 28픽셀 : 28^2 = 784)
#print(t_train.shape)  # 훈련 레이블   # 60000,
#print(x_test.shape)   # 시험 이미지   # 10000, 784   (10000개의 이미지를 시험해보자)
#print(t_test.shape)   # 시험 레이블   # 10000        (10000개의 결과)
#
# img = x_train[0]
# label = t_train[0]
# print(label)
#
# print(img.shape)
# img = img.reshape(28, 28)   ## 받아올 때 flatten = True 로 받았기 때문에 (784, ) 짜리 일차원 배열. 따라서 reshape 로 이미지 화
# print(img.shape)
# showImage(img)

# 입력층 뉴런을 784개, 출력층 뉴런을 10개 (0~9)로 구성


x, t = getData()
network = initNetwork()
accuracyCnt = 0

for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:   # 채점!
        accuracyCnt += 1


print("Accuracy : " + str(float(accuracyCnt) / len(x)))

