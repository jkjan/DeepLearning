import sys, os
sys.path.append(os.pardir)
from mnist import load_mnist
import numpy as np
from PIL import Image

def imgShow(img) :
    pilImg = Image.fromarray(np.unit8(img))
    pilImg.show()

(xTrain, tTrain), (xTest, tTest) = load_mnist(flatten=True, normalize=False)

img = xTrain[0]
label = tTrain[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)
