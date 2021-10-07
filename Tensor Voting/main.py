import cv2
from tensor_voting import *
import numpy as np
from PIL import Image

img = cv2.imread("../images/square.jpg", cv2.IMREAD_GRAYSCALE)
size = np.shape(img)



pix = iterative_tv(img, 45, 3, 0.2, 50)

pixV = np.reshape(pix, (size[0] * size[1]))

new = Image.new("L", (size[1], size[0]), 255)
new.putdata(pixV)
new.save('../output/6.jpg')