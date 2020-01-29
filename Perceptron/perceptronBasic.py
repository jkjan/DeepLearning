import numpy as np


def AND(x1, x2):   # AND perceptron
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp > theta:   # 노드*가중치의 합이 임계값 (theta) 를 넘어야
        return True   # 1 을 반환 (뉴런 활성화)
    else:
        return False


def biasedAND(x1, x2):   # AND perceptron with bias
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # 편향, bias
    tmp = np.sum(w*x) + b
    if tmp > 0:   # 노드*가중치의 합이 임계값 (theta) 를 넘어야
        return True   # 1 을 반환 (뉴런 활성화)
    else:
        return False

# w 는 가중치, 입력 신호 (노드, x) 가 결과에 주는 영향력을 조절.
# b 는 편향, 얼마나 쉽게 활성화 (1 출력) 되는지를 조정.
# b 가 -0.1 이라면 w * x 의 합이 0.1 을 넘어야 활성화된다. -20.0 이라면 20을 넘어야 한다.
# 앞으로는 임계값을 편향으로 쓰기로 한다.


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 이는 NAND 퍼셉트론으로, AND 에서의 가중치의 반대이다.
    b = 0.7
    temp = np.sum(x*w) + b
    if temp > 0:
        return True
    else:
        return False


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(x * w) + b
    if temp > 0:
        return True
    else:
        return False


# print(NAND(0, 0))
# print(NAND(0, 1))
# print(NAND(1, 0))
# print(NAND(1, 1))
