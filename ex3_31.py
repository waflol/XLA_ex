import cv2
import numpy as np
import matplotlib.pyplot as plt

# solarization
def quantimage(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img

img = cv2.imread('data/Rainbow.jpg')
img = cv2.resize(img,(512,512))
HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask = np.zeros(img.shape, dtype=np.uint8)
# roi_corners = np.array(points, dtype=np.int32)
After_img = cv2.cvtColor(quantimage(HSV_img,k=15),cv2.COLOR_HSV2BGR)
cv2.imshow('Origin image',img)
cv2.imshow('HSV',HSV_img)
cv2.imshow('After',After_img)
cv2.waitKey(0)
cv2.destroyAllWindows()