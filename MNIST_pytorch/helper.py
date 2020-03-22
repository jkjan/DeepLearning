from PIL import Image
import numpy as np

def showImage(img):
    img = img.reshape(28, 28)  ## 받아올 때 flatten = True 로 받았기 때문에 (784, ) 짜리 일차원 배열. 따라서 reshape 로 이미지 화
    pil_img = Image.fromarray(np.uint8(img))  # 배열로 이미지 불러오기
    pil_img.show()
