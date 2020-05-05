import matplotlib.pyplot as plt
import numpy as np

def show_img(img, label):
    img = img/2 + 0.5
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(label)
    plt.show()

