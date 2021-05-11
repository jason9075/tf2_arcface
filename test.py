import numpy as np
import tensorflow as tf
from convert_tensorflow import create_training_model
from test_train import ArcFaceModel

IMAGE_SIZE = (112, 112)
MODEL_TYPE = 'mobilenetv3l'
H5_WEIGHT = 'saved_model/20210427_mb3l_peter_epoch_192.h5'


def from_h5_model():
    model = create_training_model(IMAGE_SIZE, 1, mode='infer', model_type=MODEL_TYPE)

    model.load_weights(H5_WEIGHT, by_name=True, skip_mismatch=True)

    import cv2
    img1 = cv2.imread(
        'dataset/divide/astra_3b61c629-d3ea-488b-885a-5ea9c005e700/astra_3b61c629-d3ea-488b-885a-5ea9c005e700_0f29982084022362c7cd9ae7a1c76570.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, IMAGE_SIZE)
    img1 = img1 - 127.5
    img1 = img1 * 0.0078125

    vector1 = model.predict(np.expand_dims(img1, axis=0))[0]
    print(f'Person A\'s vector: \n{vector1[:10]}')

    img2 = cv2.imread(
        'dataset/divide/astra_3b61c629-d3ea-488b-885a-5ea9c005e700/astra_3b61c629-d3ea-488b-885a-5ea9c005e700_6705ef699ad23b41dc9583b58e533f56.jpg')
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


def from_opensource():
    model = ArcFaceModel(size=112,
                         num_classes=1,
                         embd_shape=512,
                         training=False)
    model.summary(line_length=80)

    model.load_weights(H5_WEIGHT, by_name=True)

    import cv2
    img1 = cv2.imread(
        'dataset/divide/astra_3b61c629-d3ea-488b-885a-5ea9c005e700/astra_3b61c629-d3ea-488b-885a-5ea9c005e700_0f29982084022362c7cd9ae7a1c76570.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, IMAGE_SIZE)
    img1 = img1 - 127.5
    img1 = img1 * 0.0078125

    vector1 = model.predict(np.expand_dims(img1, axis=0))[0]
    print(f'Person A\'s vector: \n{vector1[:10]}')

    img2 = cv2.imread(
        'dataset/divide/astra_3b61c629-d3ea-488b-885a-5ea9c005e700/astra_3b61c629-d3ea-488b-885a-5ea9c005e700_1a7653b89285a64e2f4c98e2282d5a51.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, IMAGE_SIZE)
    img2 = img2 - 127.5
    img2 = img2 * 0.0078125

    vector2 = model.predict(np.expand_dims(img2, axis=0))[0]
    print(f'Person A\'s another vector: \n{vector2[:10]}')

    sim = cosine_dist(vector1, vector2)
    print(f'same person sim: {sim}')

    img2 = cv2.imread(
        'dataset/divide/astra_d03d905e-947e-4653-8ddf-63d09795e6c1/astra_d03d905e-947e-4653-8ddf-63d09795e6c1_46d011b4bff64787acf2f31c778863e8.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, IMAGE_SIZE)
    img2 = img2 - 127.5
    img2 = img2 * 0.0078125

    vector2 = model.predict(np.expand_dims(img2, axis=0))[0]
    print(f'Person B\'s vector: \n{vector2[:10]}')

    sim = cosine_dist(vector1, vector2)
    print(f'diff person sim: {sim}')


if __name__ == '__main__':
    # from_h5_model()
    # from_saved_model()
    from_opensource()
