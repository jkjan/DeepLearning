from commonlyused import *
import numpy as np

class SimpleNet :
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        return crossEntropyError(y, t)

f = lambda w : net.loss(x, t)

net = SimpleNet()
print("랜덤 가중치")
print(net.W)
print()

x = np.array([0.6, 0.9])
p = net.predict(x)
print("예측 결과")
print(p)
print()

print("예상 답")
print(np.argmax(p))
print()

t = np.array([0, 0, 1])
print("실제 답 2와의 손실")
print(net.loss(x, t))
print()

print("기울기 함수")
dW = numericalGradient(f, net.W)
print(dW)
print()