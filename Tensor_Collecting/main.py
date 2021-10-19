import cv2
from tensor_voting import *
from tools import visualize_direction
import numpy as np
from PIL import Image

img = cv2.imread("../images/square.jpg", cv2.IMREAD_GRAYSCALE)
size = np.shape(img)



pix, vec = iterative_tv(img, 45, 3, 0.01, 55)

#visualize_direction(size[0], size[1], 1, vec)

pixV = np.reshape(pix, (size[0] * size[1]))

new = Image.new("L", (size[1], size[0]), 255)
new.putdata(pixV)
new.save('square.jpg')

 