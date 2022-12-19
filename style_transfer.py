#Style transfer 
import functools
import os

from matplotlib import gridspec
import IPython.display as display
import matplotlib.pylab as plt
import numpy as np
import PIL.Image
import tensorflow as tf
import tensorflow_hub as hub

#import user image 
! wget https://wallpapers.com/images/file/gold-snowy-christmas-tree-outside-m95abzflruoojajb.jpg -O user_image.jpg

#import style  image 
! wget https://wallpapers.com/images/file/gold-snowy-christmas-tree-outside-m95abzflruoojajb.jpg -O style_image.jpg


#since we are using tensors, need a function to convert back to images for display 
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

#load image with maximum dimensions of 512 
def loadimage(path_to_img):
  max_dim = 512
  image = tf.io.read_file(path_to_img)
  image = tf.image.decode_image(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)

  shape = tf.cast(tf.shape(image)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  image = tf.image.resize(image, new_shape)
  image = image[tf.newaxis, :]
  return image

#load images 
user_image = loadimage('user_image.jpg')
style_image = loadimage('style_image.jpg')

#display image 
def displayimage(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def styletransfer(user_image, style_image):
  # Load Hub module.
  import tensorflow_hub as hub
  hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
  #create image 
  user_image = loadimage('user_image.jpg')
  style_image = loadimage('style_image.jpg')
  styletransfer_image = hub_model(tf.constant(user_image), tf.constant(style_image))[0]
  tensor_to_image(styletransfer_image)
  displayimage(styletransfer_image)

styletransfer('user_image.jpg', 'style_image.jpg')
