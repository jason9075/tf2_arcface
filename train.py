import datetime
import os
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from dotenv import load_dotenv
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from convert_tensorflow import create_training_model

load_dotenv()

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_CLASS_NAMES = np.array([])

EPOCHS = int(os.getenv("EPOCHS"))
# IMAGE_SIZE = (224, 224)
IMAGE_SIZE = (112, 112)
RETRAIN = False

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
TRAIN_RECORD_PATH = os.getenv("TRAIN_RECORD_PATH")
VALID_RECORD_PATH = os.getenv("VALID_RECORD_PATH")
MODEL_TYPE = 'resnet50'  # (resnet50|se_resnet50)
RESTORE_FILE_PATH = 'saved_model/insightface.h5'

image_feature_description = {
    'image/source_id': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}


def main():
    train_main_ds = tf.data.TFRecordDataset(TRAIN_RECORD_PATH)
    valid_main_ds = tf.data.TFRecordDataset(VALID_RECORD_PATH)

    train_main_ds = train_main_ds.map(_parse_image_aug_function, num_parallel_calls=AUTOTUNE)
    train_main_ds = train_main_ds.shuffle(buffer_size=2000)
    train_main_ds = train_main_ds.repeat()
    train_main_ds = train_main_ds.batch(BATCH_SIZE)

    valid_main_ds = valid_main_ds.map(_parse_image_function, num_parallel_calls=AUTOTUNE)
    valid_main_ds = valid_main_ds.repeat()
    valid_main_ds = valid_main_ds.batch(BATCH_SIZE)

    num_of_class = 20000
    num_of_train_images = 663118
    num_of_valid_images = 60000
    steps_per_epoch = num_of_train_images // BATCH_SIZE + 1
    valid_steps_per_epoch = num_of_valid_images // BATCH_SIZE + 1
    model = create_training_model(IMAGE_SIZE, [3, 4, 6, 3], num_of_class, mode='train', model_type=MODEL_TYPE)

    model = restore_weight(model, restore_latest_ckpt=False, h5_filepath=RESTORE_FILE_PATH)
    model.summary()

    radam = tfa.optimizers.RectifiedAdam(LEARNING_RATE)
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    model.compile(optimizer=ranger, loss=softmax_loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    training_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    checkpoint = ModelCheckpoint(
        f"checkpoints/{training_date}_e_{{epoch}}.ckpt",
        save_freq='epoch',
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)

    record = TensorBoard(log_dir='tensorboard/',
                         update_freq='epoch',
                         profile_batch=0)

    record._total_batches_seen = steps_per_epoch
    record._samples_seen = num_of_train_images

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=[checkpoint, record],
              validation_data=valid_main_ds,
              validation_steps=valid_steps_per_epoch,
              )


def restore_weight(model, restore_latest_ckpt, h5_filepath):
    if restore_latest_ckpt:
        ckpt_path = tf.train.latest_checkpoint('checkpoints/')
        model.load_weights(ckpt_path)
    else:
        model.load_weights(h5_filepath, by_name=True)

    return model


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


def _parse_image_aug_function(example_proto):
    data = tf.io.parse_single_example(example_proto, image_feature_description)
    img = data['image/encoded']
    label = data['image/source_id']

    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)

    return (img, label), label


def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logist
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
    return tf.reduce_mean(ce)


if __name__ == '__main__':
    main()
