import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from test_train import ArcFaceModel

MEMBER_RECORD = '20000_dataset_cc.tfrecord'
VALID_RECORD = '20000_dataset_aug.tfrecord'
MIN_SIM_THR = 0.8
MODEL_TYPE = 'res50v2'
H5_WEIGHT = 'saved_model/20210514_res50v2_peter_epoch_40.h5'
IMAGE_SIZE = (112, 112)
ID_LIMIT = 5000
MEMBER_PICKLE_FILE = f'member_list_{ID_LIMIT}.pickle'
VALID_PICKLE_FILE = f'valid_list_{ID_LIMIT}.pickle'

class Member:
    def __init__(self, class_id, vector):
        self.class_id = class_id
        self.vector = vector


def main():
    image_feature_description = {
        'image/source_id': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        return tf.io.parse_single_example(example_proto, image_feature_description)

    # model = create_training_model(IMAGE_SIZE, 1, mode='infer', model_type='resnet50')
    model = ArcFaceModel(size=112,
                         num_classes=1,
                         embd_shape=512,
                         training=False,
                         model_type=MODEL_TYPE)

    model.load_weights(H5_WEIGHT, by_name=True, skip_mismatch=True)

    print('Gen Member Part:')
    if os.path.isfile(MEMBER_PICKLE_FILE):
        with open(MEMBER_PICKLE_FILE, 'rb') as handle:
            member_list = pickle.load(handle)
    else:
        member_main_ds = tf.data.TFRecordDataset(MEMBER_RECORD)
        parsed_image_dataset = member_main_ds.map(_parse_image_function)

        # Gen member list #
        member_list = []
        for data in tqdm(parsed_image_dataset):
            source_id = data['image/source_id'].numpy()
            if ID_LIMIT < source_id:
                continue
            image_raw = data['image/encoded'].numpy()
            img = cv2.imdecode(np.asarray(bytearray(image_raw)), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            img = img - 127.5
            img = img * 0.0078125

            vector = model.predict(np.expand_dims(img, axis=0))[0]
            member_list.append(Member(source_id, vector))

        with open(MEMBER_PICKLE_FILE, 'wb') as handle:
            pickle.dump(member_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Gen Valid Part:')
    if os.path.isfile(VALID_PICKLE_FILE):
        with open(VALID_PICKLE_FILE, 'rb') as handle:
            valid_list = pickle.load(handle)
    else:

        valid_main_ds = tf.data.TFRecordDataset(VALID_RECORD)
        # Compare member list #
        parsed_image_dataset = valid_main_ds.map(_parse_image_function)
        valid_list = []

        for data in tqdm(parsed_image_dataset):
            source_id = data['image/source_id'].numpy()
            if ID_LIMIT < source_id:
                continue
            image_raw = data['image/encoded'].numpy()
            img = cv2.imdecode(np.asarray(bytearray(image_raw)), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (112, 112))
            img = img - 127.5
            img = img * 0.0078125

            vector = model.predict(np.expand_dims(img, axis=0))[0]
            valid_list.append(Member(source_id, vector))

        with open(VALID_PICKLE_FILE, 'wb') as handle:
            pickle.dump(valid_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Compare Part:')
    correct_count = 0
    incorrect_count = 0
    not_found = 0
    for valid in tqdm(valid_list):
        max_sim = 0
        max_sim_id = -1

        for member in member_list:
            sim = cosine_dist(member.vector, valid.vector)

            if max_sim < sim:
                max_sim = sim
                max_sim_id = member.class_id

        if max_sim < MIN_SIM_THR:
            not_found += 1
            continue

        if max_sim_id == valid.class_id:
            correct_count += 1
        else:
            incorrect_count += 1

    print(f'correct:{correct_count}')
    print(f'incorrect_count:{incorrect_count}')
    print(f'not_found:{not_found}')


def cosine_dist(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    sim = dot / norm  # between [-1, +1]
    sim = (sim + 1) / 2

    return sim


if __name__ == '__main__':
    main()
