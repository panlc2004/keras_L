from keras.layers import Input, Dense
from keras.models import Model
import tensorflow as tf
import numpy as np

inputs = Input(shape=(7,), name='input')
x1 = Dense(12, activation='relu', name='x1')(inputs)
x2 = Dense(8, activation='relu', name='x2')(x1)
x3 = Dense(1, activation='sigmoid', name='x3')(x2)

print(inputs)