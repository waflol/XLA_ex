import cv2
import numpy as np
import matplotlib.pyplot as plt

A = cv2.imread('data/apple.jpg')
A = cv2.resize(A,(256,256))
B = cv2.imread('data/orange.jpg')
B = cv2.resize(B,(256,256))

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    L = cv2.subtract(gpA[i-1],GE)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    L = cv2.subtract(gpB[i-1],GE)
    lpB.append(L)
    
# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    ls = np.hstack((la[:,0:cols//2], lb[:,cols//2:]))
    LS.append(ls)
# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])
# image with direct connecting each half
real = np.hstack((A[:,:cols//2],B[:,cols//2:]))

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.title('Pyramid_blending2')
plt.imshow(cv2.cvtColor(ls_,cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(122)
plt.title('Direct_blending')
plt.imshow(cv2.cvtColor(real,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.savefig('data/Pyramidblending_result_ex3_20.jpg')

