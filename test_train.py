import os
import math

import numpy as np
import tensorflow as tf
from dotenv import load_dotenv
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard

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
MODEL_TYPE = 'efficientnet_b4'
# RESTORE_FILE_PATH = 'saved_model/origin_insightface.h5'

image_feature_description = {
    'image/source_id': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
}


def _regularizer(weights_decay=5e-4):
    return tf.keras.regularizers.l2(weights_decay)


def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logit
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
    return tf.reduce_mean(ce)


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


def _parse_tfrecord():
    def parse_tfrecord(tfrecord):
        features = {'image/source_id': tf.io.FixedLenFeature([], tf.int64),
                    'image/filename': tf.io.FixedLenFeature([], tf.string),
                    'image/encoded': tf.io.FixedLenFeature([], tf.string)}
        x = tf.io.parse_single_example(tfrecord, features)
        x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)

        y_train = tf.cast(x['image/source_id'], tf.float32)

        x_train = _transform_images()(x_train)
        y_train = _transform_targets(y_train)
        return (x_train, y_train), y_train

    return parse_tfrecord


def _transform_images():
    def transform_images(x_train):
        x_train = tf.image.resize(x_train, (128, 128))
        x_train = tf.image.random_crop(x_train, (112, 112, 3))
        x_train = tf.image.random_flip_left_right(x_train)
        x_train = tf.image.random_saturation(x_train, 0.6, 1.4)
        x_train = tf.image.random_brightness(x_train, 0.4)
        x_train = x_train / 255
        return x_train

    return transform_images


def _transform_targets(y_train):
    return y_train


def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle=True, buffer_size=10240):
    """load dataset from tfrecord"""
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def main():
    train_main_ds = load_tfrecord_dataset(TRAIN_RECORD_PATH, BATCH_SIZE)
    valid_main_ds = load_tfrecord_dataset(VALID_RECORD_PATH, BATCH_SIZE)

    num_of_class = 10
    num_of_train_images = 22
    num_of_valid_images = 30
    steps_per_epoch = num_of_train_images // BATCH_SIZE + 1
    valid_steps_per_epoch = num_of_valid_images // BATCH_SIZE + 1

    model = ArcFaceModel(size=112,
                         num_classes=num_of_class,
                         embd_shape=512,
                         training=True)
    model.summary(line_length=80)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)
    loss_fn = SoftmaxLoss()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model.load_weights(os.path.join('saved_model', '20210423_mb3l_peter_epoch_100.h5'), by_name=True)
    print('load weight complete.')

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              validation_data=valid_main_ds,
              validation_steps=valid_steps_per_epoch,
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
                 w_decay=5e-4, use_pretrain=True, training=False, model_type='default'):
    """Arc Face Model"""
    x = inputs = tf.keras.layers.Input([size, size, channels], name='input_image')

    x = Backbone(use_pretrain=use_pretrain, model_type=model_type)(x)

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

    def get_config(self):
        return {}


if __name__ == '__main__':
    main()
