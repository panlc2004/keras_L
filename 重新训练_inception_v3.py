# WEIGHTS_PATH = 'D:/GitCode/python/keras_L/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
import glob
import os

from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


# 数据准备
IM_WIDTH, IM_HEIGHT = 299, 299  # InceptionV3指定的图片尺寸
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 172  # 冻结层的数量

train_dir = 'D:/retrain/retrain_data_color/backpack/train_data/train_data'  # 训练集数据
val_dir = 'D:/retrain/retrain_data_color/backpack/train_data/validation_data'  # 验证集数据
# nb_classes = 12
nb_epoch = 1
batch_size = 16

nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
nb_epoch = int(nb_epoch)  # epoch数量
batch_size = int(batch_size)

# 　图片生成器
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# 训练数据与测试数据
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size, class_mode='categorical')


# 添加新层
def add_new_last_layer(base_model, nb_classes):
    """
    添加最后的层
    输入
    base_model和分类数量
    输出
    新的keras的model
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  # new FC layer, random init
    predictions = Dense(nb_classes, activation='softmax')(x)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model


optimizer = SGD(lr=0.2, momentum=0.9)


# optimizer = Adam(lr=0.1)


# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# 定义网络框架
base_model = InceptionV3(weights='imagenet', include_top=False)  # 预先要下载no_top模型
print('base_model.layers:', len(base_model.layers))
model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
setup_to_transfer_learn(model, base_model)  # 冻结base_model所有层

model.load_weights('color_weight_3.h5', by_name=True)

# 模式一训练
history_tl = model.fit_generator(
    train_generator,
    nb_epoch=nb_epoch,
    samples_per_epoch=nb_train_samples,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

model.save_weights('color_weight_3.h5')
model.save('color_3.h5')
