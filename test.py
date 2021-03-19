import numpy as np
import tensorflow as tf
from convert_tensorflow import create_training_model

IMAGE_SIZE = (112, 112)
MODEL_TYPE = 'mobilenetv3'
H5_WEIGHT = 'saved_model/20210318_mb3_epoch_400.h5'


def from_h5_model():
    model = create_training_model(IMAGE_SIZE, 1, mode='infer', model_type=MODEL_TYPE)

    model.load_weights(H5_WEIGHT, by_name=True, skip_mismatch=True)

    import cv2
    img1 = cv2.imread(
        'dataset/divide/astra_463844fe-887b-4032-9086-c976de4a24bf/astra_463844fe-887b-4032-9086-c976de4a24bf_aa0c157021ea438db44991629f35e4d0.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, IMAGE_SIZE)
    img1 = img1 - 127.5
    img1 = img1 * 0.0078125

    vector1 = model.predict(np.expand_dims(img1, axis=0))[0]
    print(f'Person A\'s vector: \n{vector1[:10]}')

    img2 = cv2.imread(
        'dataset/divide/astra_463844fe-887b-4032-9086-c976de4a24bf/astra_463844fe-887b-4032-9086-c976de4a24bf_f505cfa9fe98a564d173f95c58926c43.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, IMAGE_SIZE)
    img2 = img2 - 127.5
    img2 = img2 * 0.0078125

    vector2 = model.predict(np.expand_dims(img2, axis=0))[0]
    print(f'Person A\'s another vector: \n{vector2[:10]}')

    sim = cosine_dist(vector1, vector2)
    print(f'same person sim: {sim}')

    img2 = cv2.imread(
        'dataset/divide/astra_7ede8f23-bfd6-46d0-b06e-89445cf255d8/astra_7ede8f23-bfd6-46d0-b06e-89445cf255d8_0a2a859731dc4205dfd0f7ceeeb6888a.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, IMAGE_SIZE)
    img2 = img2 - 127.5
    img2 = img2 * 0.0078125

    vector2 = model.predict(np.expand_dims(img2, axis=0))[0]
    print(f'Person B\'s vector: \n{vector2[:10]}')

    sim = cosine_dist(vector1, vector2)
    print(f'diff person sim: {sim}')


def cosine_dist(v1, v2):
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    sim = dot / norm  # between [-1, +1]
    sim = (sim + 1) / 2

    return sim


def from_saved_model():
    model = tf.saved_model.load('saved_model/recognition/aws/')

    import cv2
    img1 = cv2.imread(
        'dataset/divide/astra_463844fe-887b-4032-9086-c976de4a24bf/astra_463844fe-887b-4032-9086-c976de4a24bf_aa0c157021ea438db44991629f35e4d0.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, IMAGE_SIZE)
    img1 = img1 - 127.5
    img1 = img1 * 0.0078125
    img1 = img1.astype(np.float32)

    vector1 = model(np.expand_dims(img1, axis=0))[0]
    print(vector1)


if __name__ == '__main__':
    from_h5_model()
    # from_saved_model()
