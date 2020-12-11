import glob
import itertools
import math
import os
import random
import shutil
from shutil import copyfile

import cv2
import tensorflow as tf
import numpy as np

from aug_func import random_blur, remap, shear_image, slight_rotate, distort_face, random_brightness_and_contrast
from face_align import align

TARGET_FOLDER = 'dataset/folder/'
MERGE_FOLDER = 'dataset/merge/'
DIVIDE_FOLDER = 'dataset/divide/'

DUP_CHECK_FOLDER = 'dataset/cele_tiny_dup/'
AUG_FOLDER = 'dataset/test_dup/'

SIZE = 224
AUG_COUNT = 3


def merge_all_file(align_face=False):
    if align_face:
        landmarks_detector = tf.keras.models.load_model('tf_model/face_landmark/1/')

    for img_path in glob.glob(os.path.join(TARGET_FOLDER, '*', '*', '*.jpg')):
        if not img_path.endswith('.jpg'):
            continue

        path_split = img_path.split('/')
        org = path_split[-3]
        member_name = path_split[-2]
        file_name = path_split[-1]

        # right_landmarks = np.array([
        #     [0.31, 0.21],
        #     [0.71, 0.25],
        #     [0.65, 0.50],
        #     [0.33, 0.73],
        #     [0.70, 0.73]], dtype=np.float32)
        # right_landmarks *= SIZE

        if align_face:
            origin_image = cv2.imread(img_path)
            h, w, _ = origin_image.shape
            img = cv2.resize(origin_image, (SIZE, SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.expand_dims(img, axis=0)
            img = img - 127.5
            img = img * 0.0078125

            landmarks = landmarks_detector.predict(img)[0]
            # draw_landmark(origin_image, landmarks.copy(), h, w)
            landmarks = np.asarray([(int(landmark[0] * w),
                                     int(landmark[1] * h))
                                    for landmark in landmarks])

            img = align(origin_image, landmarks, SIZE)

            # for r in right_landmarks:
            #     px = int(r[0])
            #     py = int(r[1])
            #     cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
            cv2.imwrite(os.path.join(MERGE_FOLDER, f'{org}_{member_name}_{file_name}'), img)
        else:
            copyfile(img_path, os.path.join(MERGE_FOLDER, f'{org}_{member_name}_{file_name}'))


def divide_folder():
    shutil.rmtree(DIVIDE_FOLDER, ignore_errors=True)

    os.mkdir(DIVIDE_FOLDER)

    for image_path in glob.glob(os.path.join(MERGE_FOLDER, '*.jpg')):
        file_name = image_path.split('/')[-1]
        member_name = '%s_%s' % (file_name.split('_')[-3], file_name.split('_')[-2])
        if not os.path.isdir(os.path.join(DIVIDE_FOLDER, member_name)):
            os.mkdir(os.path.join(DIVIDE_FOLDER, member_name))
        copyfile(image_path, os.path.join(DIVIDE_FOLDER, member_name, file_name))


def draw_landmark(origin_img, result, h, w):
    result[:, 0] = result[:, 0] * w
    result[:, 1] = result[:, 1] * h
    result = result.astype(int)

    for r in result:
        cv2.circle(origin_img, tuple(r), 3, (0, 255, 0), -1)


def check_dup_face(check_times=1):
    vector_predictor = tf.keras.models.load_model('tf_model/face_recognition')
    dirs = os.listdir(DUP_CHECK_FOLDER)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(DUP_CHECK_FOLDER, d))]

    similarity_set = set()
    for _ in range(check_times):
        before_add_count = len(similarity_set)
        vector_dict = {}

        # select one of image each folder
        for member_name in dirs:
            member_img_paths = glob.glob(os.path.join(DUP_CHECK_FOLDER, member_name, '*.jpg'))

            chosen_one = random.choice(member_img_paths)

            img = cv2.imread(chosen_one)
            img = cv2.resize(img, (112, 112))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img - 127.5
            img = img * 0.0078125
            vector = vector_predictor.predict(np.expand_dims(img, axis=0))[0]
            vector_dict[chosen_one] = vector

        # compare each dist
        for a, b in itertools.combinations(vector_dict, 2):
            v1, v2 = vector_dict[a], vector_dict[b]
            dist = np.linalg.norm(v1 - v2)

            if dist < 0.5:
                similarity_set.add((a.split('/')[-2], b.split('/')[-2]))

        print(f'Add {len(similarity_set) - before_add_count} of samples.')

    print(similarity_set)


def aug_image(img, aug_order):
    done_aug = set()
    for aug_index in aug_order:
        if aug_index == 0:
            continue
        elif aug_index == 1:
            if aug_index in done_aug:
                continue
            img = random_blur(img)
            done_aug.add(aug_index)
        elif aug_index == 2:
            if aug_index in done_aug:
                continue
            img = distort_face(img)
            done_aug.add(aug_index)
        elif aug_index == 3:
            img = shear_image(img)
        elif aug_index == 4:
            img = slight_rotate(img)

    return img

def random_light(img):

    return img


def aug_data():
    dirs = os.listdir(DUP_CHECK_FOLDER)
    dirs = [d for d in dirs if os.path.isdir(os.path.join(DUP_CHECK_FOLDER, d))]
    dirs = [d for d in dirs if not d.startswith('.')]

    for member_name in dirs[:10]:
        if os.path.isdir(os.path.join(AUG_FOLDER, member_name)):
            continue

        os.mkdir(os.path.join(AUG_FOLDER, member_name))

        member_img_paths = glob.glob(os.path.join(DUP_CHECK_FOLDER, member_name, '*.jpg'))
        for idx in range(AUG_COUNT):
            chosen_one = random.choice(member_img_paths)

            img = cv2.imread(chosen_one)

            aug_order = [random.randint(0, 4) for _ in range(3)]
            img = aug_image(img, aug_order)
            img = random_brightness_and_contrast(img)

            cv2.imwrite(os.path.join(AUG_FOLDER, member_name, f'{member_name}-aug{idx}.jpg'), img)
            del img



def main():
    # merge_all_file(align_face=True)
    # divide_folder()
    # check_dup_face(check_times=10)
    aug_data()


if __name__ == '__main__':
    main()
