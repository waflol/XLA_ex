import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from cv2.ximgproc import guidedFilter
import os,sys
import random

def addition_noise(img):
    rw,cl,dept = img.shape
    noise_img = img.copy()
    num_pixls = random.randint(300,20000)
    for i in range(num_pixls):
        y_crd = random.randint(0,rw-1)
        x_crd = random.randint(0,cl-1)
        noise_img[x_crd][y_crd] = 255
    
    for i in range(num_pixls):
        y_crd = random.randint(0,rw-1)
        x_crd = random.randint(0,cl-1)
        noise_img[x_crd][y_crd] = 0
    return noise_img


def bilinear_fitering(img,d=0,sigmaColor=0, sigmaSpace=0):
    bilateral_blur = cv2.bilateralFilter(img,d,sigmaColor,sigmaSpace)
    return bilateral_blur

def guided_filtering(guide,img,r = 2,eps = 0.05):
    return guidedFilter(guide = guide, src = img, radius = r, eps = eps)


img = plt.imread('data/flower.jpg')
img = cv2.resize(img,(256,256))
noise_img = addition_noise(img)
plt.figure(figsize=(16,16))
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(noise_img)


sigmas = [5, 10, 20]
# Bilinear filtering
plot_id = 234
plt.figure(figsize=(16,16))
for sigma in sigmas:
    smooth_img = bilinear_fitering(img = img,d = sigma,sigmaColor=80, sigmaSpace=80)
    plt.subplot(plot_id)
    plt.title("Bilinear Filtering  ($r$=%s)" %sigma)
    plt.imshow(smooth_img)
    plt.axis('off')
    plot_id +=1

# Guided filtering
plot_id = 234
plt.figure(figsize=(16,16))
for sigma in sigmas:
    smooth_img = guided_filtering(guide = noise_img,img = img,r = sigma,eps = 0.05)
    plt.subplot(plot_id)
    plt.title("Guided Filtering ($r$=%s)" %sigma)
    plt.imshow(smooth_img)
    plt.axis('off')
    plot_id +=1
    
    
plt.show()
    

    

