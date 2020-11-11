import pathlib
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint

from backend.utils import load_weight
from loss_func.loss import ArcMarginPenalty
from model.se_resnet50 import create_se_resnet50

AUTOTUNE = tf.data.experimental.AUTOTUNE
TRAIN_CLASS_NAMES = np.array([])

TRAIN_DATA_PATH = 'dataset/train/'
VALID_DATA_PATH = 'dataset/val/'
EPOCHS = 50
IMAGE_SIZE = (112, 112)

BATCH_SIZE = 32


def softmax_loss(y_true, y_pred):
    # y_true: sparse target
    # y_pred: logist
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
    return tf.reduce_mean(ce)


def create_training_model(input_shape, layers, num_of_class, embedding_size=128,
                          training=False, margin=0.5, logit_scale=64):
    input_node = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))

    net = create_se_resnet50(input_node, layers)

    if training:
        labels = tf.keras.layers.Input([], name='labels')

        net = tf.keras.layers.Dropout(0.4)(net)
        net = tf.keras.layers.Flatten()(net)
        pre_logits = tf.keras.layers.Dense(embedding_size)(net)

        logits = ArcMarginPenalty(num_classes=num_of_class, margin=margin,
                                  logit_scale=logit_scale, embedding_size=embedding_size)((pre_logits, labels))
        model = tf.keras.Model(inputs=[input_node, labels], outputs=[logits])

    else:
        model = tf.keras.Model(inputs=[input_node], outputs=[net])

    model.summary()

    return model


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

    load_weight(model, 'pytorch_pretrained/se_resnet50-ce0d4300.pth', trainable=False, verbose=False)
    model.compile(optimizer="Adam", loss=softmax_loss)

    checkpoint = ModelCheckpoint(
        'checkpoints/e_{epoch}_b_{batch}.ckpt',
        save_freq=int(steps_per_epoch * 10), verbose=1,
        save_weights_only=True)

    model.fit(train_main_ds,
              epochs=EPOCHS,
              steps_per_epoch=steps_per_epoch,
              callbacks=[checkpoint],
              # initial_epoch=epochs - 1)
              )


def get_label(file_path):
    parts = tf.strings.split(file_path, '/')
    one_hot = tf.cast(parts[-2] == TRAIN_CLASS_NAMES, tf.float32)
    return tf.argmax(one_hot), one_hot


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


def dummy():
    from trident import senet
    from trident import load_lfw
    senet.SE_ResNet50(include_top=False, pretrained=True, input_shape=(3, 112, 112)).with_optimizer().with_loss()
    load_lfw(format='aligned_face', is_paired=False)


if __name__ == '__main__':
    main()
