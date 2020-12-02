import math
import tensorflow as tf


class ArcMarginPenalty(tf.keras.layers.Layer):

    def __init__(self, num_classes, margin=0.5, logit_scale=64, embedding_size=128, **kwargs):
        super(ArcMarginPenalty, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.margin = margin
        self.logit_scale = logit_scale
        self.embedding_size = embedding_size
        self.w = None
        self.cos_m = None
        self.sin_m = None
        self.th = None
        self.mm = None

    def build(self, input_shape):
        self.w = self.add_variable(
            "weights", shape=[self.embedding_size, self.num_classes])
        self.cos_m = tf.identity(math.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(math.sin(self.margin), name='sin_m')
        self.th = tf.identity(math.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    @tf.function
    def call(self, inputs, **kwargs):
        embds, labels = inputs
        normed_embds = tf.nn.l2_normalize(embds, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(
            cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logist = tf.where(mask == 1., cos_mt, cos_t)
        logist = tf.multiply(logist, self.logit_scale, 'arcface_logist')

        return logist
