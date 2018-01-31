from keras import Sequential, Input, Model
from keras.layers import Dense, Convolution2D, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.datasets import cifar10
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

batch_size = 32
nb_classes = 10
nb_epoch = 20
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

inputs = Input(shape=X_train.shape[1:])
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

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')

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

    # 计算特征方向归一化所需的数量
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        epochs=nb_epoch,
                        validation_data=(X_test, Y_test))


