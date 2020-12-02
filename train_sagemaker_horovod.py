import datetime
import os
import pathlib
from argparse import ArgumentParser

import horovod.keras as hvd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras.optimizers import Adam

from convert_tensorflow import create_training_model

tf.compat.v1.disable_eager_execution()

parser = ArgumentParser()
parser.add_argument('--batch_size', default=128, help='batch_size')
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

args = parser.parse_args()

TRAIN_DATA_PATH = os.path.join(input_path, 'training')
EPOCHS = int(args.epoch)
IMAGE_SIZE = (int(args.image_size), int(args.image_size))
BATCH_SIZE = int(args.batch_size)
FREQ_FACTOR = int(args.freq_factor_by_number_of_epoch)

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def main():
    global TRAIN_CLASS_NAMES

    hvd_size = hvd.size()

    train_data_dir = pathlib.Path(TRAIN_DATA_PATH)
    train_list_ds = tf.data.Dataset.list_files(str(train_data_dir / '*/*.jpg'))
    TRAIN_CLASS_NAMES = np.array(
        [item.name for item in train_data_dir.glob('*') if item.name not in [".keep", ".DS_Store"]])
    num_of_class = len(TRAIN_CLASS_NAMES)
    train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
    steps_per_epoch = np.ceil(train_image_count // BATCH_SIZE // hvd_size)

    train_main_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    train_main_ds = prepare_for_training(train_main_ds)

    model = create_training_model(IMAGE_SIZE, [3, 4, 6, 3], num_of_class, training=True)

    # download pre trained weight
    import boto3
    s3 = boto3.client('s3')
    s3.download_file('astra-face-recognition-dataset',
                     f'pretrained/{args.pretrained}',
                     os.path.join('saved_model', args.pretrained))

    model.load_weights(os.path.join('saved_model', args.pretrained), by_name=True)
    model.summary()

    # radam = tfa.optimizers.RectifiedAdam(0.001 * hvd_size)
    # ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    adam = Adam(0.001 * hvd_size)
    adam = hvd.DistributedOptimizer(adam)

    model.compile(optimizer=adam, loss=softmax_loss)
    training_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    callbacks = []

    callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    callbacks.append(hvd.callbacks.MetricAverageCallback())
    callbacks.append(hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=5, verbose=1))
    if hvd.rank() == 0:
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
