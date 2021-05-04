import os
import math
import datetime

import tensorflow as tf
from argparse import ArgumentParser
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

from backend.callbacks import MaxCkptSave

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
parser.add_argument('--lr', default=0.1, help='learning rate')
parser.add_argument('--freeze_layers', default=0, help='freeze num of last layers')

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
LR = float(args.lr)
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
FREEZE_LAYERS = int(args.freeze_layers)

image_feature_description = {
    'image/source_id': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}

PRETRAIN_BUCKET = 'sagemaker-us-east-1-astra-face-recognition'


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logit
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
        ds = ds.repeat()
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.map(_dataset_parser_train, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(BATCH_SIZE)
    else:
        ds = ds.repeat()
        ds = ds.map(_dataset_parser_valid,
                    num_parallel_calls=AUTOTUNE)
        ds = ds.batch(VALID_BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds


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

    train_main_ds = prepare_for_training(train_main_ds)
    steps_per_epoch = np.ceil(TRAIN_IMAGE_COUNT / BATCH_SIZE)

    valid_main_ds = PipeModeDataset(channel='valid', record_format='TFRecord')
    valid_main_ds = prepare_for_training(valid_main_ds, is_train=False)
    valid_steps_per_epoch = np.ceil(VALID_IMAGE_COUNT / VALID_BATCH_SIZE)

    with strategy.scope():
        model = ArcFaceModel(size=112,
                             num_classes=NUM_CLASSES,
                             embd_shape=512,
                             training=True)

        if args.pretrained != 'None':
            import boto3
            s3 = boto3.client('s3')
            s3.download_file(PRETRAIN_BUCKET,
                             f'pretrained/{args.pretrained}',
                             os.path.join('saved_model', args.pretrained))
            model.load_weights(os.path.join('saved_model', args.pretrained), by_name=True)
            print('load weight complete.')

        model.summary(line_length=80)

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=LR, momentum=0.9, nesterov=True)
        loss_fn = SoftmaxLoss()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

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
              verbose=VERBOSE,
              validation_freq=1,
              callbacks=callbacks,
              )


def SoftmaxLoss():
    """softmax loss"""

    def softmax_loss(y_true, y_pred):
        # y_true: sparse target
        # y_pred: logist
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)
        return tf.reduce_mean(ce)

    return softmax_loss


def ArcFaceModel(size=None, channels=3, num_classes=None, name='arcface_model',
                 margin=0.5, logist_scale=64, embd_shape=512,
                 w_decay=5e-4, use_pretrain=True, training=False):
    """Arc Face Model"""
    x = inputs = tf.keras.layers.Input([size, size, channels], name='input_image')

    x = Backbone(use_pretrain=use_pretrain, model_type=MODEL_TYPE)(x)

    embds = OutputLayer(embd_shape, w_decay=w_decay)(x)

    if training:
        assert num_classes is not None
        labels = tf.keras.layers.Input([], name='label')
        logist = ArcHead(num_classes=num_classes, margin=margin,
                         logist_scale=logist_scale)(embds, labels)
        return tf.keras.Model((inputs, labels), logist, name=name)
    else:
        return tf.keras.Model(inputs, embds, name=name)


def ArcHead(num_classes, margin=0.5, logist_scale=64, name='ArcHead'):
    """Arc Head"""

    def arc_head(x_in, y_in):
        x = inputs1 = tf.keras.layers.Input(x_in.shape[1:])
        y = tf.keras.layers.Input(y_in.shape[1:])
        x = ArcMarginPenaltyLogists(num_classes=num_classes,
                                    margin=margin,
                                    logist_scale=logist_scale)(x, y)
        return tf.keras.Model((inputs1, y), x, name=name)((x_in, y_in))

    return arc_head


def Backbone(use_pretrain=True, model_type='default'):
    """Backbone Model"""
    def backbone(x_in):
        if model_type == 'default':
            return tf.keras.applications.MobileNetV2(input_shape=x_in.shape[1:], include_top=False)(x_in)
        elif model_type == 'mobilenetv3l':
            return tf.keras.applications.MobileNetV3Large(input_shape=x_in.shape[1:], include_top=False)(x_in)
        elif model_type == 'res50v2':
            return tf.keras.applications.ResNet50V2(input_shape=x_in.shape[1:], include_top=False)(x_in)
        elif model_type == 'efficientnet_b4':
            return tf.keras.applications.EfficientNetB4(input_shape=x_in.shape[1:], include_top=False)(x_in)
        else:
            raise RuntimeError(f'model type \'{model_type}\' not exist.')

    return backbone


def OutputLayer(embd_shape, w_decay=5e-4, name='OutputLayer'):
    """Output Later"""

    def output_layer(x_in):
        x = inputs = tf.keras.layers.Input(x_in.shape[1:])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(rate=0.5)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(embd_shape, kernel_regularizer=_regularizer(w_decay))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return output_layer


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """Make trainable=False freeze BN for real (the og version is sad).
       ref: https://github.com/zzh8829/yolov3-tf2
    """

    def call(self, x, training=False):
        if training is None:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super().call(x, training)


class ArcMarginPenaltyLogists(tf.keras.layers.Layer):
    """ArcMarginPenaltyLogists"""

    def __init__(self, num_classes, margin=0.5, logist_scale=64, **kwargs):
        super(ArcMarginPenaltyLogists, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[int(input_shape[-1]), self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, embds, labels):
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logists = tf.where(mask == 1., cos_mt, cos_t)
        logists = tf.multiply(logists, self.logist_scale, 'arcface_logist')

        return logists


if __name__ == '__main__':
    main()
