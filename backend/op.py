import tensorflow as tf


class NormDense(tf.keras.layers.Layer):

    def __init__(self, classes=1000):
        super(NormDense, self).__init__()
        self.classes = classes
        self.w = None

    def build(self, input_shape):
        self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.classes),
                                 initializer='random_normal', trainable=True)

    def call(self, inputs, **kwargs):
        norm_w = tf.nn.l2_normalize(self.w, axis=0)
        x = tf.matmul(inputs, norm_w)

        return x


def squeeze_excite(x, se_filters, num_filters, use_bias=True, name=''):
    out = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Reshape((1, 1, x.shape[-1]))(out)
    out = tf.keras.layers.Conv2D(se_filters, (1, 1), activation='relu', use_bias=use_bias,
                                 name=f'{name}.se_module.fc1__convbias')(out)
    out = tf.keras.layers.Conv2D(num_filters, (1, 1), activation='sigmoid', use_bias=use_bias,
                                 name=f'{name}.se_module.fc2__convbias')(out)

    return x * out


def se_bottleneck(x, num_filters, strides=1, expansion=4, conv_shortcut=True, name=''):
    shortcut = x
    if 1 < strides or conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(num_filters * expansion,
                                          (1, 1),
                                          strides=strides,
                                          use_bias=False,
                                          padding='same',
                                          name=f'{name}.downsample.0__conv')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{name}.downsample.1__bn')(shortcut)

    out = tf.keras.layers.Conv2D(num_filters,
                                 (1, 1),
                                 strides=strides,
                                 use_bias=False,
                                 padding='same',
                                 name=f'{name}.conv1__conv')(x)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{name}.bn1__bn')(out)
    out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)

    out = tf.keras.layers.Conv2D(num_filters,
                                 (3, 3),
                                 strides=1,
                                 use_bias=False,
                                 padding='same',
                                 name=f'{name}.conv2__conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{name}.bn2__bn')(out)
    out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)

    out = tf.keras.layers.Conv2D(num_filters * expansion,
                                 (1, 1),
                                 strides=1,
                                 use_bias=False,
                                 padding='same',
                                 name=f'{name}.conv3__conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{name}.bn3__bn')(out)
    out = tf.keras.layers.ReLU(name=f'{name}.relu3')(out)

    out = squeeze_excite(out, num_filters // expansion, num_filters * expansion, use_bias=True, name=name)

    return tf.keras.layers.ReLU(name=f'{name}.relu4')(shortcut + out)
