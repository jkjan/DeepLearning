import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))


print("신경망을 구현해보자.")

X = np.array([1.0, 0.5])   # 0층, (2x1) 행렬
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 가중치
B1 = np.array([0.1, 0.2, 0.3])   # 편향

print(W1.shape)  # (2, 3)
print(X.shape)   # (2,)
print(B1.shape)  # (3,)
print()

print("이때 우리가 활성화 함수에 넣어줄 값 A = X * W1 + B 로 계산됨.")
print("X 가 브로드캐스팅 돼서 1x2 (가로로 긴) 행렬로 바뀌어 1x2 x 2x3  행렬 간의 곱")
print("곱셈 결과의 1x3 행렬에 이후 편향 B1 을 더한다.")
print("결과는 a(2층) 의 활성화 함수에 넣어줄 값 세 개 (3,)")

A1 = np.dot(X, W1) + B1

print(A1.shape)
print(A1)
print()

print("이때 활성화 함수가 등장. 활성화 함수로는 시그모이드 함수를 사용.")

Z1 = sigmoid(A1)

print("Z1 = ", Z1)
print()


print("여기까지가 x1 에서 a(1) 로 가는 과정 (0층 -> 1층, 입력층에서 은닉층으로 도입)")
print()
print("1층에서 2층으로 : ")

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print("Z1 shape = ", Z1.shape)
print("W2 shape = ", W2.shape)
print("B2 shape = ", B2.shape)
print()

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print("Z2 = ", Z2)
print()


print("2층에서 3층, 마지막층으로 : ")

def identityFunction(x): # 항등 함수
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identityFunction(A3)

print("Final result = ", A3)

