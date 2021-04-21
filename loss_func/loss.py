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
        self.w = self.add_weight(
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

        # method origin #
        # cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')
        # sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')
        #
        # cos_mt = tf.subtract(
        #     cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')
        #
        # cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)
        #
        # mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
        #                   name='one_hot_mask')
        #
        # logist = tf.where(mask == 1., cos_mt, cos_t)
        # logist = tf.multiply(logist, self.logit_scale, 'arcface_logist')

        # method 2 #
        # fc7 = tf.matmul(normed_embds, normed_w, name='fc7')
        # mapping_label_onehot = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
        #                                   name='one_hot_mask')
        # fc7_onehot = fc7 * mapping_label_onehot
        # cos_t = fc7_onehot
        # t = tf.math.acos(cos_t)
        # t = t + self.margin
        # margin_cos = tf.math.cos(t)
        # margin_fc7 = margin_cos
        # margin_fc7_onehot = margin_fc7 * mapping_label_onehot
        # diff = margin_fc7_onehot - fc7_onehot
        # fc7 = fc7 - diff
        # logist = tf.multiply(fc7, self.logit_scale, 'arcface_logist')

        # method 3 #
        m = self.margin
        s = self.logit_scale
        cos_m = math.cos(m)
        sin_m = math.sin(m)

        mm = sin_m * m

        threshold = math.cos(math.pi - m)

        cos_t = tf.matmul(normed_embds, normed_w, name='cos_t')

        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)
        keep_val = s * (cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)
        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                                          name='one_hot_mask')
        inv_mask = tf.subtract(1., mask, name='inverse_mask')
        s_cos_t = tf.multiply(cos_t, s, name='scalar_cos_t')
        logist = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_logits')

        return logist

    def get_config(self):
        return {}
