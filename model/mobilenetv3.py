import tensorflow as tf


def create_mobilenetv3(input_node):
    model = tf.keras.applications.MobileNetV3Small(input_tensor=input_node, include_top=False)

    return model.output
