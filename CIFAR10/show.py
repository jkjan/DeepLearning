import matplotlib.pyplot as plt
import numpy as np

def showImg(img):
    img = img/2 + 0.5
    npImg = img.numpy()
    plt.imshow(np.transpose(npImg, (1, 2, 0)))
    plt.show()

