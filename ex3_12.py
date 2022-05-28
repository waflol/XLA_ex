#In[] Bai 3.12
import cv2
import numpy as np
from matplotlib import pyplot as plt

def compare_image(image1, image2):
	plt.figure(figsize=(9,9))
	plt.subplot(1,2,1)
	plt.imshow(image1)
	plt.title('Orignal')
	plt.axis('off')

	plt.subplot(1,2,2)
	plt.imshow(image2)
	plt.title('Modified')
	plt.axis('off')

	plt.tight_layout()

image = cv2.imread("data/lena.png")
 
from skimage.util import random_noise
## adding noise
noise_img = random_noise(image, mode='s&p',amount=0.3)
noise_img = np.array(255*noise_img, dtype = 'uint8')
## median filter
median = cv2.medianBlur(noise_img,5)
compare_image(noise_img,median)

park = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
edge = cv2.Canny(park,100,200)
compare_image(park,edge)

img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations=1)
compare_image(img,erosion)

grad = cv2.morphologyEx(img,cv2.MORPH_GRADIENT, kernel)
compare_image(img,grad)
# %%
