import sys

from Differential import *
# 함수를 미분했을 때 기울기 벡터는 그 함수의 최솟값을 가리킨다.
# 이 방법을 써서 학습을 할 때 가중치를 조절한다.
# 이를 경사법이라고 한다.
# 경사법을 사용하면, 손실 함수의 값이 최소가 되는 방향으로 학습을 진행할 수 있다.

# 각 x, 즉 가중치를 학습 후 -(미분값 * 학습률) 를 곱하도록 하자

def gradientDescent(f, initX, lr = 0.01, stepNum = 100):
    x = initX
    for i in range(stepNum):
        grad = numericalGradient(f, x)
        x -= lr * grad
    return x

initX = np.array([-3.0, 4.0])
arr = gradientDescent(function2, initX = initX, lr = 0.1, stepNum=100)
print(arr)

print(sys.path)