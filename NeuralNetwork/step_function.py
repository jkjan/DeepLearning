import numpy as np
import matplotlib.pylab as plt

x = np.array([-1.0, 1.0, 2.0])
y = x > 0

print(y)  # 넘파이 배열에 부등호 연산을 수행하면
          # 원소 각각에 부등호 연산을 수행한 bool 배열을 반환한다.
print(y.astype(np.int))  # astype 함수는 매개변수에 명시된 자료형으로 배열 원소 각각을 캐스팅


# 계단 함수
def stepFunction(x):
    return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1)
y = stepFunction(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
