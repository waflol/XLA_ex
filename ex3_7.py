import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from skimage.io import imshow, imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte, img_as_float
from skimage.exposure import histogram, cumulative_distribution

def show(img):
    cv2.imshow('',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
img = cv2.imread('data/Rainbow.jpg')
img = cv2.resize(img,(1024,1024))
show(img)

#ep 1
XYZ_img = cv2.cvtColor(img,cv2.COLOR_RGB2XYZ)
show(XYZ_img)


#ep2
histSize = 256
histRange = (0, 256) # the upper boundary is exclusive

r_hist = cv2.calcHist(img, [0], None, [histSize], histRange, accumulate=False)
g_hist = cv2.calcHist(img, [1], None, [histSize], histRange, accumulate=False)
b_hist = cv2.calcHist(img, [2], None, [histSize], histRange, accumulate=False)

plt.title("Histogram of all RGB Colors")
plt.plot(b_hist, color="blue")
plt.plot(g_hist, color="green")
plt.plot(r_hist, color="red")
plt.show()

r_hist_accumulate = cv2.calcHist(img, [0], None, [histSize], histRange, accumulate=True)
g_hist_accumulate = cv2.calcHist(img, [1], None, [histSize], histRange, accumulate=True)
b_hist_accumulate = cv2.calcHist(img, [2], None, [histSize], histRange, accumulate=True)

plt.title("Histogram of all RGB Colors")
plt.plot(b_hist_accumulate, color="blue")
plt.plot(g_hist_accumulate, color="green")
plt.plot(r_hist_accumulate, color="red")
plt.show()



