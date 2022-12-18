# Blur filter with user input 
#import image 
! wget https://wallpapers.com/images/file/gold-snowy-christmas-tree-outside-m95abzflruoojajb.jpg -O user_image.jpg
#read image 
from imageio import imread
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = imread('user_image.jpg')

#filter image 
def blurFilter (image,blur_amount):
  blur_image = cv2.GaussianBlur(image, (blur_amount,blur_amount), 0)
  plt.figure(figsize=(18, 8)) 
  plt.subplot(121)
  plt.imshow(image)
  plt.title('input')
  plt.axis('off')
  plt.subplot(122)
  plt.imshow(blur_image)
  plt.title('blurred image')
  plt.axis('off')
 

blurFilter(image, 55)
