import cv2
import matplotlib.pyplot as plt
import numpy as np


def ImagePad_func(img,mode='zero',num_pad = 2,**kwargs):
    if mode == 'zero':
        return np.pad(img,((num_pad,num_pad),(num_pad,num_pad),(0,0)),mode='constant',constant_values=0)
    elif mode == 'constant':
        pad_value = kwargs.get('padder', 255)
        return np.pad(img,((num_pad,num_pad),(num_pad,num_pad),(0,0)),mode='constant',constant_values=pad_value)
    elif mode == 'wrap':
        return np.pad(img,((num_pad,num_pad),(num_pad,num_pad),(0,0)),mode='wrap')
    elif mode == 'clamp':
        return np.pad(img,((num_pad,num_pad),(num_pad,num_pad),(0,0)),mode='edge')
    elif mode == 'mirror':
        return np.pad(img,((num_pad,num_pad),(num_pad,num_pad),(0,0)),mode='reflect')

def show_image(img,tab_name=''):
    cv2.imshow(tab_name,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Đọc ảnh
img = plt.imread('data/Rainbow.jpg')
img = cv2.resize(img,(512,512))

zero_pad = ImagePad_func(img,num_pad=100)
show_image(zero_pad,'zero pad')
cons_pad = ImagePad_func(img,num_pad=100,mode='constant',padder=100)
show_image(cons_pad,'constant pad')
wrap_pad = ImagePad_func(img,num_pad=100,mode='wrap')
show_image(wrap_pad,'wrap pad')
clamp_pad = ImagePad_func(img,num_pad=100,mode='clamp')
show_image(clamp_pad,'clamp pad')
mirror_pad = ImagePad_func(img,num_pad=100,mode='mirror')
show_image(mirror_pad,'mirror pad')



"""
The advantages:

The disadvantages:

"""
