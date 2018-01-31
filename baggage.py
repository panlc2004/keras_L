import numpy as np
from keras import Input, Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from DataLoader import find_data_by_batch, batch_img_reshape

nb_classes = 12
img_width = 150
img_height = 150
batch_size = 16
train_data_dir = 'D:/retrain/retrain_data_color/backpack/train_data/train_data'
validation_data_dir = 'D:/retrain/retrain_data_color/backpack/train_data/validation_data'


folder_path = 'D:/retrain/retrain_data_color/backpack/train_data'
(x_train_batch, y_train_batch), (x_test, y_test) = find_data_by_batch(folder_path, batch_size, 0.2)

inputs = Input(shape=(150, 150, 3))
x = Conv2D(32, (3, 3), padding='same')(inputs)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding='same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes)(x)
predictions = Activation('softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# adam = Adam(lr=0.1, decay=1e-3)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# 这将做预处理和实时数据增加
datagen = ImageDataGenerator(
    featurewise_center=False,  # 在数据集上将输入平均值设置为0
    samplewise_center=False,  # 将每个样本均值设置为0
    featurewise_std_normalization=False,  # 将输入除以数据集的std
    samplewise_std_normalization=False,  # 将每个输入除以其std
    zca_whitening=False,  # 应用ZCA白化
    rotation_range=0,  # 在一个范围下随机旋转图像(degrees, 0 to 180)
    width_shift_range=0.1,  # 水平随机移位图像（总宽度的分数）
    height_shift_range=0.1,  # 随机地垂直移动图像（总高度的分数）
    horizontal_flip=True,  # 随机翻转图像
    vertical_flip=False)  # 随机翻转图像


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    samples_per_epoch=2000 // batch_size,
    nb_epoch=20,
    validation_data=validation_generator,
    nb_val_samples=2000 // batch_size)