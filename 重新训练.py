from keras.applications.inception_resnet_v2 import InceptionResNetV2
import tensorflow as tf
from keras.utils import plot_model


with tf.variable_scope("InceptionResNetV2"):
    model = InceptionResNetV2(include_top=True, weights=None)

model.summary()

print(model.output)
# plot_model(model, to_file='model.png')
# model.get_layer('Dense_1').output

