import tensorflow as tf


def create_mobilenetv3(input_node, embedding_size=512, is_train=False, dropout=0.25):
    if is_train:
        model = tf.keras.applications.MobileNetV3Small(input_tensor=input_node, include_top=False, dropout_rate=dropout)
    else:
        model = tf.keras.applications.MobileNetV3Small(input_tensor=input_node, include_top=False)

    net = tf.keras.layers.Dense(embedding_size, name='fc__fc')(model.output)

    return net
