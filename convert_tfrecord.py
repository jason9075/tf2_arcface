import glob
import os
import random
from collections import Set, OrderedDict
from itertools import count

import cv2
import tensorflow as tf
import tqdm

TOTAL_IDS = 10
DATA_PATH = 'dataset/divide_aug'
output_name = DATA_PATH.split('/')[-1]
OUTPUT_PATH = os.path.join('dataset', f'{TOTAL_IDS}_{output_name}.tfrecord')


class IndexOrderedSet(Set):
    """An OrderedFrozenSet-like object
       Allows constant time 'index'ing
       But doesn't allow you to remove elements"""

    def __init__(self, iterable=()):
        self.num = count()
        self.dict = OrderedDict(zip(iterable, self.num))

    def add(self, elem):
        if elem not in self:
            self.dict[elem] = next(self.num)

    def index(self, elem):
        return self.dict[elem]

    def __contains__(self, elem):
        return elem in self.dict

    def __len__(self):
        return len(self.dict)

    def __iter__(self):
        return iter(self.dict)

    def __repr__(self):
        return 'IndexOrderedSet({})'.format(self.dict.keys())


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def make_example(img_str, source_id, filename):
    feature = {'image/source_id': _int64_feature(source_id),
               'image/filename': _bytes_feature(filename),
               'image/encoded': _bytes_feature(img_str)}

    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    id_set = IndexOrderedSet()
    samples = []
    id_folders = os.listdir(DATA_PATH)
    id_dict = {}

    if TOTAL_IDS != -1:
        id_folders = id_folders[:TOTAL_IDS]

    id_count = 0
    for id_name in tqdm.tqdm(id_folders):
        id_dict[id_name] = id_count
        img_paths = glob.glob(os.path.join(DATA_PATH, id_name, '*.jpg'))
        for img_path in img_paths:
            id_set.add(id_name)
            filename = os.path.join(id_name, os.path.basename(img_path))
            samples.append((img_path, id_name, filename))

        id_count += 1
    random.shuffle(samples)

    with tf.io.TFRecordWriter(OUTPUT_PATH) as writer:
        for img_path, id_name, filename in tqdm.tqdm(samples):
            img = cv2.imread(img_path)

            cv2.imwrite("temp.jpg", img)
            tf_example = make_example(img_str=open("temp.jpg", 'rb').read(),
                                      source_id=id_dict[id_name],
                                      filename=str.encode(filename))
            writer.write(tf_example.SerializeToString())

    print(f'Total ids: {len(id_set)}')
    print(f'Total samples: {len(samples)}')


def test():
    raw_dataset = tf.data.TFRecordDataset('dataset/10_divide.tfrecord')
    image_feature_description = {
        'image/source_id': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
        image_raw = image_features['image/source_id'].numpy()


if __name__ == '__main__':
    # main()
    test()
