import tensorflow as tf

from loss_func.loss import ArcMarginPenalty
from model.se_resnet50 import create_se_resnet50

AUTOTUNE = tf.data.experimental.AUTOTUNE

IMAGE_SIZE = (224, 224)


def create_training_model(input_shape, layers, num_of_class, embedding_size=128,
                          mode='infer', margin=0.5, logit_scale=64):
    input_node = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], 3))

    net = create_se_resnet50(input_node, layers)

    if mode == 'train':
        labels = tf.keras.layers.Input([], name='labels')

        net = tf.keras.layers.Dropout(0.4)(net)
        net = tf.keras.layers.Flatten()(net)
        pre_logits = tf.keras.layers.Dense(embedding_size, use_bias=False)(net)

        logits = ArcMarginPenalty(num_classes=num_of_class, margin=margin,
                                  logit_scale=logit_scale, embedding_size=embedding_size)((pre_logits, labels))
        model = tf.keras.Model(inputs=[input_node, labels], outputs=[logits])

    elif mode == 'infer':
        net = tf.keras.layers.Flatten()(net)
        net = tf.keras.layers.Dense(embedding_size, use_bias=False)(net)
        net = tf.nn.l2_normalize(net, axis=1, name='normed_embd')

        model = tf.keras.Model(inputs=[input_node], outputs=[net])
    else:
        raise ValueError('model must equal (train|infer)')

    return model


def main():
    model = create_training_model(IMAGE_SIZE, [3, 4, 6, 3], 1, mode='infer')

    from backend.utils import load_weight
    load_weight(model, 'pytorch_pretrained/se_resnet50-ce0d4300.pth', trainable=False, verbose=False)
    model.summary()

    model.save("saved_model/tf_pretrain_weight.h5", include_optimizer=False, save_format='h5')


if __name__ == '__main__':
    main()
