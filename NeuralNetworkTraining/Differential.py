import numpy as np
import matplotlib.pylab as plt

# f 라는 수식을 x 에 대해서 미분해보자.
def numericalDiff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h))/(h*2)

# 이는 수치 미분이라 한다. 수식을 직접 미분해서 대입할 수 없으니
# 임의로 아주 작은 값 (0.0001) 을 넣어주는 것이다.

# 0.01x^2 + 0.1x 이라는 함수를 만들어보자.
def function1(x):
    return 0.01*x**2 + 0.1*x

x = np.arange(0.0, 20.0, 0.1)
y = function1(x)

print(numericalDiff(function1, 5))
print(numericalDiff(function1, 10))


# 다음은 f(x0, x1) = x0^2 + x1^2 , x 가 두 개인 함수이다.
def function2(x):
    return np.sum(x**2)

# 편미분은 수학적으로 하나의 변수를 고정한 뒤 그에 대해 미분한다.
# 동시에 계산할 때는, 각 미분 결과에 따른 것을 벡터로 나타낸다.
def numericalGradient(f, x):
    h = 1e-4
    gradient = np.zeros_like(x)

    for i in range(x.size):
        tmpVal = x[i]
        x[i] = tmpVal + h    # 각 원소에 대해서 수치 미분
        fxh1 = f(x)
        x[i] = tmpVal - h
        fxh2 = f(x)

        gradient[i] = (fxh1 - fxh2) / (2*h) # 각 결과를 gradient 라는 벡터에 담는다
        x[i] = tmpVal

    return gradient

print(numericalGradient(function2, np.array([3.0, 4.0])))
print(numericalGradient(function2, np.array([0.0, 2.0])))
print(numericalGradient(function2, np.array([3.0, 0.0])))

# 이때 이 기울기는 함수의 가장 낮은 장소 (최소값) 을 가리킨다.
# 즉 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향이다.

