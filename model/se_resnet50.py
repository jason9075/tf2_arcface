import tensorflow as tf

from backend.op import se_bottleneck


def create_se_resnet50(input_node, layers):
    def _make_layer(x, num_filters, block_layers, strides=1, name=''):
        conv_shortcut = False
        if strides != 1 or num_filters != 128:
            conv_shortcut = True

        out = se_bottleneck(x, num_filters, strides=strides, expansion=4, conv_shortcut=conv_shortcut, name=f'{name}.0')

        for k in range(1, block_layers):
            out = se_bottleneck(out, num_filters, strides=1, expansion=4, conv_shortcut=False, name=f'{name}.{k}')

        return out

    net = tf.keras.layers.Conv2D(64,
                                 (7, 7),
                                 strides=2,
                                 use_bias=False,
                                 padding='same',
                                 name='layer0.conv1__conv')(input_node)
    net = tf.keras.layers.BatchNormalization(epsilon=1e-5, name='layer0.bn1__bn')(net)
    net = tf.keras.layers.ReLU(name='layer0.relu1')(net)

    net = _make_layer(net, 64, layers[0], strides=1, name='layer1')
    net = _make_layer(net, 128, layers[1], strides=2, name='layer2')
    net = _make_layer(net, 256, layers[2], strides=2, name='layer3')
    net = _make_layer(net, 512, layers[3], strides=2, name='layer4')
    net = tf.keras.layers.GlobalAveragePooling2D()(net)

    return net

