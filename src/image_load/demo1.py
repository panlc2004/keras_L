import matplotlib.pyplot as plot
import keras
from keras_preprocessing import image
import numpy as np

img_path = 'F:\\facenet_train_data\\train\\person152\\person152_5.png'
img = image.load_img(img_path, target_size=(196,196))

# 要除以255.0， 要不然图片是灰色
img = image.img_to_array(img) / 255.
img = np.expand_dims(img, axis=0)
print(img.shape)
img1 = img.astype('float32')
plot.imshow(img[0])
plot.show()



