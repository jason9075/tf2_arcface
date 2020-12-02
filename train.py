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

TRAIN_DATA_PATH = os.getenv("TRAIN_DATA_PATH")
VALID_DATA_PATH = os.getenv("VALID_DATA_PATH")
EPOCHS = int(os.getenv("EPOCHS"))
IMAGE_SIZE = (224, 224)
RETRAIN = False

BATCH_SIZE = int(os.getenv("BATCH_SIZE"))


def main():
    global TRAIN_CLASS_NAMES

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    train_list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
    TRAIN_CLASS_NAMES = np.array(
        [item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    num_of_class = len(TRAIN_CLASS_NAMES)
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    steps_per_epoch = np.ceil(train_image_count / BATCH_SIZE)

    train_main_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_main_ds = prepare_for_training(train_main_ds)

    model = create_training_model(IMAGE_SIZE, [3, 4, 6, 3], num_of_class, training=True)

    if RETRAIN:
        ckpt_path = tf.train.latest_checkpoint('checkpoints/')
        model.load_weights(ckpt_path)
    else:
        model.load_weights('saved_model/tf_pretrain_weight.h5', by_name=True)
    model.summary()

    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

    model.compile(optimizer=ranger, loss=softmax_loss)
    training_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    checkpoint = ModelCheckpoint(
        f"checkpoints/{training_date}_e_{{epoch}}.ckpt",
        save_freq=int(steps_per_epoch * 10), verbose=1,
        save_best_only=True,
        save_weights_only=True)

    record = TensorBoard(log_dir='saved_model/',
                         update_freq=1,
                         profile_batch=0)

    record._total_batches_seen = steps_per_epoch
    record._samples_seen = steps_per_epoch * BATCH_SIZE

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=[checkpoint, record],
              # initial_epoch=epochs - 1)
              )


def decode_img(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)
    return img


def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logist
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
    return tf.reduce_mean(ce)


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == TRAIN_CLASS_NAMES, tf.float32)
    return tf.argmax(one_hot), one_hot


def process_path(file_path):
    label, _ = get_label(file_path)
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return (img, label), label


def prepare_for_training(ds, cache=False, shuffle_buffer_size=2000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


if __name__ == '__main__':
    main()
