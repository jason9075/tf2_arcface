import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from convert_tensorflow import create_training_model
from train import softmax_loss

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_CLASS_NAMES = np.array([])

BATCH_SIZE = 32
num_of_class = 20000
IMAGE_SIZE = (224, 224)
RETRAIN = False


def main():
    eval_dataset = tf.data.TFRecordDataset('dataset/20000_dataset_aug.tfrecord')
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

    model = create_training_model(IMAGE_SIZE, num_of_class, mode='train')

    if RETRAIN:
        ckpt_path = tf.train.latest_checkpoint('saved_model/ckpt-20210104/')
        model.load_weights(ckpt_path)
    else:
        model.load_weights('saved_model/2021-01-04-03-48-31_e_10.h5')
    model.summary()

    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    model.compile(optimizer=ranger, loss=softmax_loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.evaluate(eval_dataset)


if __name__ == '__main__':
    main()
