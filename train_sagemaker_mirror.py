import datetime
import os
import pathlib
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from convert_tensorflow import create_training_model

parser = ArgumentParser()
parser.add_argument('--batch_size', default=16, help='batch_size')
parser.add_argument('--epoch', default=3, help='epoch')
parser.add_argument('--freq_factor_by_number_of_epoch', default=1, help='freq_factor_by_number_of_epoch')
parser.add_argument('--image_size', default=224, help='image_size')
parser.add_argument('--model_dir', default="", help='model_dir')
parser.add_argument('--pretrained', default="", help='pretrained')
parser.add_argument('--task_name', default="fr-train", help='task_name')

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
ckpt_path = os.path.join(prefix, 'model', 'ckpt')
tb_path = os.path.join(prefix, 'model', 'tensorboard')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_CLASS_NAMES = np.array([])

# with open(param_path, 'r') as tc:
#     trainingParams = json.load(tc)

strategy = tf.distribute.MirroredStrategy()
print("MirroredStrategy REPLICAS: ", strategy.num_replicas_in_sync)

args = parser.parse_args()

TRAIN_DATA_PATH = os.path.join(input_path, 'train')
EPOCHS = int(args.epoch)
IMAGE_SIZE = (int(args.image_size), int(args.image_size))
BATCH_SIZE = int(args.batch_size) * strategy.num_replicas_in_sync
FREQ_FACTOR = int(args.freq_factor_by_number_of_epoch)


gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def main():
    global TRAIN_CLASS_NAMES

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    train_list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
    TRAIN_CLASS_NAMES = np.array(
        [item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    num_of_class = len(TRAIN_CLASS_NAMES)
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    steps_per_epoch = np.ceil(train_image_count // BATCH_SIZE)

    train_main_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_main_ds = prepare_for_training(train_main_ds)

    with strategy.scope():
        model = create_training_model(IMAGE_SIZE, num_of_class, mode='train', model_type='mobilenetv3')

    # model.summary()

    adam = Adam(0.01)

    model.compile(optimizer=adam, loss=softmax_loss)
    training_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    callbacks = []

    callbacks.append(ModelCheckpoint(
        os.path.join(ckpt_path, f"{training_date}_e_{{epoch}}"),
        save_freq=int(steps_per_epoch * FREQ_FACTOR)))
    callbacks.append(TensorBoard(log_dir=tb_path,
                                 update_freq=int(steps_per_epoch * FREQ_FACTOR),
                                 profile_batch=0))

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks,
              verbose=1
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
