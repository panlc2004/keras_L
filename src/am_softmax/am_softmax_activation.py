from keras.engine.base_layer import Layer
from keras.layers import activations
from keras.layers import initializers
from keras.layers import regularizers
from keras.layers import constraints
from keras.layers import interfaces
from keras.layers import Dense
from keras.engine.base_layer import InputSpec
import keras.backend as K


class AmSoftmax(Dense):

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        w = self.add_weight(shape=(input_dim, self.units),
                            initializer=self.kernel_initializer,
                            name='kernel',
                            regularizer=self.kernel_regularizer,
                            constraint=self.kernel_constraint)
        self.kernel = w / (K.epsilon() + K.sqrt(K.sum(K.square(w),
                                                      axis=0,
                                                      keepdims=True)))
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
