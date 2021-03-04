import tensorflow as tf


def create_mobilenetv2(input_node):
    model = tf.keras.applications.MobileNetV2(input_tensor=input_node, include_top=False, weights=None)

    return model.output
