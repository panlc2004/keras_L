import glob
import os

from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

NB_IV3_LAYERS_TO_FREEZE = 172
IM_WIDTH, IM_HEIGHT = 299, 299  # InceptionV3指定的图片尺寸
train_dir = 'D:/retrain/retrain_data_color/backpack/train_data/train_data'  # 训练集数据
val_dir = 'D:/retrain/retrain_data_color/backpack/train_data/validation_data'  # 验证集数据
# nb_classes = 12
nb_epoch = 20
batch_size = 16

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
nb_epoch = int(nb_epoch)  # epoch数量
batch_size = int(batch_size)


# model = InceptionV3(weights='imagenet', include_top=True)
model = load_model('total_color_test.h5')
print('model.layers:', len(model.layers))


# 冻上NB_IV3_LAYERS之前的层
def setup_to_finetune(model):
    """
    Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
    note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
    Args:
      model: keras model
    """
    for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.0001, decay=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 设置网络结构
setup_to_finetune(model)


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

model.summary()

# loss, acc = model.evaluate_generator(validation_generator)
# print('loss: ', loss, '  acc: ', acc)


# 模式二训练
history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    nb_epoch=nb_epoch,
    validation_data=validation_generator,
    nb_val_samples=nb_val_samples,
    class_weight='auto')

# 模型保存
model.save('total_color_test_2.h5')
