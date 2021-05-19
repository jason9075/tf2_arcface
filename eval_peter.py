import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from convert_tensorflow import create_training_model
from test_train import ArcFaceModel
from train import softmax_loss

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_CLASS_NAMES = np.array([])

BATCH_SIZE = 32
num_of_class = 93979
IMAGE_SIZE = (112, 112)
MODEL_TYPE = 'res50v2'
h5_filepath = 'saved_model/20210516_res50v2_peter_epoch_80.h5'
CKPT = 'checkpoints/2021-04-22-03-04-21_e_2'
TFRECORD = 'dataset/10_divide_112.tfrecord'


def main():
    eval_dataset = tf.data.TFRecordDataset(TFRECORD)
    image_feature_description = {
        'image/source_id': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        data = tf.io.parse_single_example(example_proto, image_feature_description)
        img = data['image/encoded']
        label = data['image/source_id']

        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)
        img = tf.subtract(img, 127.5)
        img = tf.multiply(img, 0.0078125)

        return (img, label), label

    eval_dataset = eval_dataset.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    eval_dataset = eval_dataset.batch(BATCH_SIZE)

    # debug_data = eval_dataset.as_numpy_iterator()
    # print(next(debug_data))

    model = ArcFaceModel(size=112,
                         num_classes=num_of_class,
                         embd_shape=512,
                         training=True,
                         model_type=MODEL_TYPE)
    model.load_weights(h5_filepath, by_name=True)
    model.summary(line_length=80)

    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    model.compile(optimizer=ranger, loss=softmax_loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.evaluate(eval_dataset)


if __name__ == '__main__':
    main()
