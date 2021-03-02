import tensorflow as tf


def create_mobilenetv2(input_node, embedding_size=512, is_train=False):
    model = tf.keras.applications.MobileNetV2(input_tensor=input_node, include_top=False)

    net = tf.keras.layers.Dense(embedding_size, name='fc__fc')(model.output)

    return net
