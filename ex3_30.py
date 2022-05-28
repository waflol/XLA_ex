import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
def addition_noise(img):
    rw,cl,dept = img.shape
    noise_img = img.copy()
    num_pixls = random.randint(300,15000)
    for i in range(num_pixls):
        y_crd = random.randint(0,rw-1)
        x_crd = random.randint(0,cl-1)
        noise_img[x_crd][y_crd] = 255
    
    for i in range(num_pixls):
        y_crd = random.randint(0,rw-1)
        x_crd = random.randint(0,cl-1)
        noise_img[x_crd][y_crd] = 0
    return noise_img

img = cv2.imread('data/lena.png')
dst = cv2.fastNlMeansDenoisingColored(img, None, 11, 6, 7, 21)
cv2.imshow('origin image',img)
cv2.imshow('denoised image',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()