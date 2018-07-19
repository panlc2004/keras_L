import keras
import time
import numpy as np

from keras_preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator1 = train_datagen.flow_from_directory(
    'F:\\facenet_train_data\\train',
    target_size=(96, 96),
    batch_size=400,
    class_mode='binary')

train_generator2 = train_datagen.flow_from_directory(
    'F:\\facenet_train_data\\train',
    target_size=(96, 96),
    batch_size=400,
    class_mode='binary')

train_generator = zip(train_generator1, train_generator2)

# samples = train_generator.samples
# print('samples: ', samples)
# print(train_generator.class_indices)

for x, y in train_generator:
    lab = []
    for i in range(len(x[1])):
        if x[1][i] == y[1][i]:
            lab.append(1)
        else:
            lab.append(0)
    print(np.sum(lab))
# a = [x[1], y[1]]
#     a = np.asarray(a)
#     print(a.shape)
#     print('===========')
