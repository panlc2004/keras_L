from keras import Model
from keras.applications import InceptionV3
from keras.layers import GlobalAveragePooling2D, Dense
from tensorflow import Tensor

base_model = InceptionV3(weights='imagenet', include_top=False)

base_model.summary()

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(200, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 248 layers and unfreeze the rest:
for layer in model.layers[:248]:
    layer.trainable = False
for layer in model.layers[248:]:
    layer.trainable = True

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# model.fit_generator