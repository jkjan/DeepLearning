import numpy as np
import matplotlib.pylab as plt
import step_function as sf

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
y2 = sf.stepFunction(x)

plt.plot(x, y2, linestyle="--", label="step")
plt.plot(x, y, label="sigmoid")
plt.legend()
plt.show()