import tensorflow as tf


def create_resnet50(input_node, layers=None, embedding_size=512, is_train=False):
    expansion = 1

    net = tf.keras.layers.ZeroPadding2D(padding=1, name='first_padding')(input_node)
    net = tf.keras.layers.Conv2D(64,
                                 3,
                                 strides=1,
                                 use_bias=False,
                                 name='conv1__conv')(net)
    net = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn1__bn')(net)
    net = tf.keras.layers.PReLU(alpha_initializer=tf.initializers.constant(0.25),
                                shared_axes=[1, 2], name='prelu__prelu')(net)

    net = make_layer(net, 64, layers[0], stride=2, expansion=expansion, prefix='layer1')
    net = make_layer(net, 128, layers[1], stride=2, expansion=expansion, prefix='layer2')
    net = make_layer(net, 256, layers[2], stride=2, expansion=expansion, prefix='layer3')
    net = make_layer(net, 512, layers[3], stride=2, expansion=expansion, prefix='layer4')

    net = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='bn2__bn')(net)

    # Because in pytorch is channel first and it start from index 1, so the pytorch order is NCHW.
    # And here we have to switch the tensorflow order from NHWC to NCHW
    net = tf.transpose(net, [0, 3, 1, 2])
    net = tf.keras.layers.Flatten()(net)

    if is_train:
        net = tf.keras.layers.Dropout(0.4)(net)

    net = tf.keras.layers.Dense(embedding_size, name='fc__fc')(net)
    net = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='features__bn')(net)

    return net


def make_layer(net, out_ch, num_layer, stride=1, expansion=1, prefix='layer'):
    net = basic_block(net, 64, out_ch, stride=stride, expansion=expansion, downsample=True, prefix=f'{prefix}.0')
    for idx in range(1, num_layer):
        net = basic_block(net, 64, out_ch, stride=1, expansion=expansion, prefix=f'{prefix}.{idx}')

    return net


def basic_block(net, in_ch,
                out_ch,
                stride=1,
                groups=1,
                expansion=1,
                downsample=False,
                prefix='basic_block'):
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.bn1__bn')(net)
    out = tf.keras.layers.ZeroPadding2D(padding=1, name=f'{prefix}.padding1')(out)
    out = tf.keras.layers.Conv2D(out_ch,
                                 3,
                                 strides=1,
                                 use_bias=False,
                                 name=f'{prefix}.conv1__conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.bn2__bn')(out)
    out = tf.keras.layers.PReLU(alpha_initializer=tf.initializers.constant(0.25),
                                shared_axes=[1, 2],
                                name=f'{prefix}.prelu__prelu')(out)
    out = tf.keras.layers.ZeroPadding2D(padding=1, name=f'{prefix}.padding2')(out)
    out = tf.keras.layers.Conv2D(out_ch,
                                 3,
                                 strides=stride,
                                 use_bias=False,
                                 name=f'{prefix}.conv2__conv')(out)
    out = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.bn3__bn')(out)

    if downsample and (stride != 1 or in_ch != out_ch * expansion):
        net = tf.keras.layers.Conv2D(out_ch * expansion, 1,
                                     strides=stride,
                                     use_bias=False,
                                     groups=groups,
                                     name=f'{prefix}.downsample.0__conv')(net)
        net = tf.keras.layers.BatchNormalization(epsilon=1e-5, name=f'{prefix}.downsample.1__bn')(net)

    return out + net
