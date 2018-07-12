import numpy as np
from keras.models import load_model

model = load_model('model.h5', compile=False)

a = np.zeros([2, 7])
a[0] = a[0] - 1
a[1] = a[1] + 1

predict = model.predict(a)
print(predict)

model.save_weights('model_weight.h5')
