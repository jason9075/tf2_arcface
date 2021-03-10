import datetime
import os
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from backend.callbacks import MaxCkptSave
from convert_tensorflow import create_training_model


parser = ArgumentParser()
parser.add_argument('--batch_size', default=16, help='batch_size')
parser.add_argument('--epoch', default=3, help='epoch')
parser.add_argument('--freq_factor_by_number_of_epoch', default=1, help='freq_factor_by_number_of_epoch')
parser.add_argument('--image_size', default=224, help='image_size')
parser.add_argument('--model_dir', default="", help='model_dir')
parser.add_argument('--pretrained', default="", help='pretrained')
parser.add_argument('--task_name', default="fr-train", help='task_name')
parser.add_argument('--num_of_class', default=353, help='num_of_class')
parser.add_argument('--train_image_count', default=6382, help='train_image_count')
parser.add_argument('--valid_image_count', default=300, help='valid_image_count')
parser.add_argument('--modeltype', default="mobilenet_v2", help='model type')
parser.add_argument('--verbose', default=2, help='verbose')
parser.add_argument('--max_ckpt', default=2, help='max save ckpt')

prefix = '/opt/ml/'

input_path = prefix + 'input/data'
ckpt_path = os.path.join(prefix, 'model', 'ckpt')
tb_path = os.path.join(prefix, 'model', 'tensorboard')
param_path = os.path.join(prefix, 'input/config/hyperparameters.json')

AUTOTUNE = tf.data.experimental.AUTOTUNE

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
strategy = tf.distribute.MirroredStrategy()
print("MirroredStrategy REPLICAS: ", strategy.num_replicas_in_sync)

args = parser.parse_args()

TRAIN_DATA_PATH = os.path.join(input_path, 'train')
EPOCHS = int(args.epoch)
IMAGE_SIZE = (int(args.image_size), int(args.image_size))
BATCH_SIZE = int(args.batch_size) * strategy.num_replicas_in_sync
VALID_BATCH_SIZE = 3 * strategy.num_replicas_in_sync
FREQ_FACTOR = int(args.freq_factor_by_number_of_epoch)
NUM_CLASSES = int(args.num_of_class)
TRAIN_IMAGE_COUNT = int(args.train_image_count)
VALID_IMAGE_COUNT = int(args.valid_image_count)
MODEL_TYPE = args.modeltype
VERBOSE = int(args.verbose)
MAX_CKPT = int(args.max_ckpt)

PRETRAIN_BUCKET = 'sagemaker-us-east-1-astra-face-recognition'


def _dataset_parser_train(value):
    featdef = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(value, featdef)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    label = tf.cast(example['image/source_id'], tf.int32)
    image = _train_preprocess_fn(image)
    return (image, label), label


def _dataset_parser_valid(value):
    featdef = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(value, featdef)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    label = tf.cast(example['image/source_id'], tf.int32)
    image = _valid_preprocess_fn(image)
    return (image, label), label


def _train_preprocess_fn(img):
    """Preprocess a single training image of layout [height, width, depth]."""
    # Resize the image to add four extra pixels on each side.
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.image.random_saturation(img, 0.6, 1.6)
    img = tf.image.random_contrast(img, 0.6, 1.4)
    img = tf.image.random_flip_left_right(img)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)

    return img


def _valid_preprocess_fn(img):
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=255.0)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img, 0.0078125)

    return img


def main():
    from sagemaker_tensorflow import PipeModeDataset
    train_main_ds = PipeModeDataset(channel='train', record_format='TFRecord')

    train_main_ds = train_main_ds.map(_dataset_parser_train,
                                      num_parallel_calls=AUTOTUNE)
    train_main_ds = prepare_for_training(train_main_ds)
    steps_per_epoch = np.ceil(TRAIN_IMAGE_COUNT / BATCH_SIZE)

    valid_main_ds = PipeModeDataset(channel='valid', record_format='TFRecord')
    valid_main_ds = valid_main_ds.map(_dataset_parser_valid,
                                      num_parallel_calls=AUTOTUNE)
    valid_main_ds = prepare_for_training(valid_main_ds, is_train=False)
    valid_steps_per_epoch = np.ceil(VALID_IMAGE_COUNT / VALID_BATCH_SIZE)

    with strategy.scope():
        model = create_training_model(IMAGE_SIZE, NUM_CLASSES, mode='train', model_type=MODEL_TYPE)

        if args.pretrained != 'None':
            import boto3
            s3 = boto3.client('s3')
            s3.download_file(PRETRAIN_BUCKET,
                             f'pretrained/{args.pretrained}',
                             os.path.join('saved_model', args.pretrained))
            model.load_weights(os.path.join('saved_model', args.pretrained), by_name=True)

        model.summary()

        adam = Adam(0.1)
        model.compile(optimizer=adam, loss=softmax_loss, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    training_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    callbacks = []

    callbacks.append(ModelCheckpoint(
        os.path.join(ckpt_path, f"{training_date}_e_{{epoch}}"),
        save_freq='epoch',
        mode='min',
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    ))
    callbacks.append(TensorBoard(log_dir=tb_path,
                                 update_freq=int(steps_per_epoch * FREQ_FACTOR),
                                 profile_batch=0))

    callbacks.append(MaxCkptSave(ckpt_path, MAX_CKPT))

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_main_ds,
              validation_steps=valid_steps_per_epoch,
              validation_freq=1,
              callbacks=callbacks,
              verbose=VERBOSE
              # initial_epoch=epochs - 1)
              )

    model.save_weights(os.path.join(ckpt_path, f"{training_date}_e_{EPOCHS}"))


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


def prepare_for_training(ds, cache=False, is_train=True, shuffle_buffer_size=2000):
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if is_train:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE)
    else:
        ds = ds.batch(VALID_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


if __name__ == '__main__':
    main()
