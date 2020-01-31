import numpy as np
from PIL import Image

def stepFunction(x):
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    x -= c
    expA = np.exp(x)
    return expA / np.sum(np.exp(x))

def meanSquaredError(y, t) :
    return np.sum((y-t)**2)/2

def crossEntropyError(y, t) :
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def showImage(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def numericalGradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad

def gradientDescent(f, initX, lr = 0.01, stepNum = 100):
    x = initX
    for i in range(stepNum):
        grad = numericalGradient(f, x)
        x -= lr * grad
    return x