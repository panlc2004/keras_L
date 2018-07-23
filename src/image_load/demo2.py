import keras.preprocessing.image as image

a = image.load_img('E:\GitCode\python\keras_L\src\image_load\\001.jpg', target_size=[96, 96])
a = image.img_to_array(a) / 255.
print(a.shape)


image.save_img('E:\GitCode\python\keras_L\src\image_load\person0_2.png', a)
