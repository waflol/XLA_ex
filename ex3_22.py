import cv2
import numpy as np
import matplotlib.pyplot as plt

ddepth = cv2.CV_16S
kernel_size = 3
src = cv2.imread('data/lena.png')
src = cv2.GaussianBlur(src, (3, 3), 0)
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
dst = cv2.Laplacian(src_gray, ddepth, ksize=kernel_size)
abs_dst = cv2.convertScaleAbs(dst)

cv2.imshow('origin image',src)
cv2.imshow('result',abs_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()