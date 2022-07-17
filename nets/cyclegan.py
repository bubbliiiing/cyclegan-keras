import keras.backend as K
from keras import constraints, initializers, layers, regularizers
from keras.layers import *
from keras.models import *


class InstanceNormalization(Layer):
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filter_num, block):
    conv_name_base  = 'res' + block + '_branch'
    in_name_base    = 'in' + block + '_branch'

    x = ZeroPadding2D((1, 1))(input_tensor)
    x = Conv2D(filter_num, kernel_size, name=conv_name_base + '2a')(x)
    x = InstanceNormalization(axis=3, name=in_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(filter_num, kernel_size, name=conv_name_base + '2c')(x)
    x = InstanceNormalization(axis=3, name=in_name_base + '2c')(x)
    #-------------------#
    #   残差网络
    #-------------------#
    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def generator(input_shape, out_channel = 3, n_residual_blocks = 9):
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))
    
    # 128,128,3 -> 128,128,64
    x = ZeroPadding2D((3, 3))(inputs)
    x = Conv2D(64, (7, 7))(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 128,128,64 -> 64,64,128
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    # 64,64,128 -> 32,32,256
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), strides=2)(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)

    for i in range(n_residual_blocks):
        x = identity_block(x, (3, 3), 256, block=str(i))
    
    # 32,32,256 -> 64,64,128
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3))(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)
    
    # 64,64,128 -> 128,128,64
    x = UpSampling2D((2, 2))(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(64, (3, 3))(x)
    x = InstanceNormalization(axis=3)(x)
    x = Activation('relu')(x)    

    # 128,128,64 -> 128,128,3
    x = ZeroPadding2D((3, 3))(x)
    x = Conv2D(out_channel, (7, 7))(x)
    x = Activation('tanh')(x)  
    
    model = Model(inputs, x)
    return model

def discriminator(input_shape):
    inputs = Input(shape=(input_shape[0], input_shape[1], 3))
    
    x = ZeroPadding2D((1, 1))(inputs)
    x = Conv2D(64, kernel_size=4, strides=2)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
        
    # 32,32,128
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, kernel_size=4, strides=2)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # 16,16,256
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, kernel_size=4, strides=2)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # 8,8,512
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, kernel_size=4)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    # 对每个像素点判断是否有效
    # 64
    # 8,8,1
    x = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs, x)
    return model

if __name__ == "__main__":
    model = discriminator(128)
    model.summary()
