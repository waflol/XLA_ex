import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import GaussianBlur

def pyrDown(tensor_img):
    gauss_blur = GaussianBlur(kernel_size=(5,5),sigma=(1/256.,36/256.))
    gauss_img = gauss_blur(tensor_img)
    # gauss_img = cv2.GaussianBlur(tensor_img[0].T.numpy(),(5,5),0)
    down_sampling = nn.Upsample(scale_factor=0.5,mode='bilinear')
    # return down_sampling(torch.tensor([gauss_img.T]))
    return down_sampling(gauss_img)

def pyrUp(tensor_img):
    up_sampling = nn.Upsample(scale_factor=2,mode='bilinear')
    return up_sampling(tensor_img)

alpha = 1.5

A = cv2.imread('data/apple.jpg')
A = cv2.resize(A,(256,256))

B = cv2.imread('data/orange.jpg')
B = cv2.resize(B,(256,256))

A = A/255.
B = B/255.

tensor_A = torch.tensor([A.T])
tensor_B = torch.tensor([B.T])

# generate Gaussian pyramid for A
G = tensor_A.clone()
gpA = [G]
for i in range(6):
    G = pyrDown(G)
    gpA.append(G)
    
# generate Gaussian pyramid for B
G = tensor_B.clone()
gpB = [G]
for i in range(6):
    G = pyrDown(G)
    gpB.append(G)


# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = pyrUp(gpA[i])
    L = torch.subtract(gpA[i-1],GE)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = pyrUp(gpB[i])
    L = torch.subtract(gpB[i-1],GE)
    lpB.append(L)
    

# Now add left and right halves of images in each level
LS = []
for la,lb in zip(lpA,lpB):
    _,dpt,rows,cols = la.shape
    ls = torch.concat((la[:,:,0:cols//2,:], lb[:,:,cols//2:,:]),dim=2)
    LS.append(ls)
    
# now reconstruct
ls_ = LS[0]
for i in range(1,6):
    ls_ = pyrUp(ls_)
    ls_ = torch.add(ls_, alpha*LS[i])
    
real = torch.concat((tensor_A[:,:,0:cols//2,:],tensor_B[:,:,cols//2:,:]),dim=2)

plt.figure(figsize=(16,16))
plt.subplot(121)
plt.title('Pyramid_blending_pytorch')
plt.imshow(ls_[0].T.numpy()[:,:,-1::-1])
plt.axis('off')

plt.subplot(122)
plt.title('Direct_blending')
plt.imshow(real[0].T.numpy()[:,:,-1::-1])
plt.axis('off')
plt.savefig('data/Pyramidblending_result_ex3_21.jpg')
plt.show()