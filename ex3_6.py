import cv2
import numpy as np
import matplotlib.pyplot as plt
# contrast adjustment
image = cv2.imread('data/input.jpg')
new_image = np.zeros(image.shape, image.dtype)
alpha = 3 # Simple contrast control
beta = 50    # Simple brightness control
# Initialize values
# print(' Basic Linear Transforms ')
# print('-------------------------')
# try:
#     alpha = float(input('* Enter the alpha value [1.0-3.0]: '))
#     beta = int(input('* Enter the beta value [0-100]: '))
# except ValueError:
#     print('Error, not a number')
    

# Do the operation new_image(i,j) = alpha*image(i,j) + beta
# Instead of these 'for' loops we could have used simply:
# new_image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
# but we wanted to show you how to access the pixels :)
# for y in range(image.shape[0]):
#     for x in range(image.shape[1]):
#         for c in range(image.shape[2]):
#             new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
            
new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
cv2.imshow('Original Image', image)
cv2.imshow('New Image', new_image)
# Wait until user press some key




# solarization
def quantimage(image,k):
    i = np.float32(image).reshape(-1,3)
    condition = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,20,1.0)
    ret,label,center = cv2.kmeans(i, k , None, condition,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    final_img = center[label.flatten()]
    final_img = final_img.reshape(image.shape)
    return final_img


cv2.imshow('quantimage k = 5',quantimage(image,5))
cv2.imshow('quantimage k = 8',quantimage(image,8))
cv2.imshow('quantimage k = 25',quantimage(image,25))
cv2.imshow('quantimage k = 35',quantimage(image,35))
cv2.imshow('quantimage k = 45',quantimage(image,45))
cv2.waitKey(0)
cv2.destroyAllWindows()