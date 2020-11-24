import numpy as np

from convert_tensorflow import create_training_model

IMAGE_SIZE = (224, 224)


def main():
    model = create_training_model(IMAGE_SIZE, [3, 4, 6, 3], 1, training=False)

    model.load_weights('checkpoints/2020-11-18-08-51-21_e_400.ckpt')

    import cv2
    img1 = cv2.imread(
        'dataset/divide/astra_463844fe-887b-4032-9086-c976de4a24bf/astra_463844fe-887b-4032-9086-c976de4a24bf_aa0c157021ea438db44991629f35e4d0.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, IMAGE_SIZE)
    img1 = img1 - 127.5
    img1 = img1 * 0.0078125

    vector1 = model.predict(np.expand_dims(img1, axis=0))[0]
    print(vector1)

    # img2 = cv2.imread('dataset/divide/astra_463844fe-887b-4032-9086-c976de4a24bf/astra_463844fe-887b-4032-9086-c976de4a24bf_f505cfa9fe98a564d173f95c58926c43.jpg')
    img2 = cv2.imread(
        'dataset/divide/astra_7ede8f23-bfd6-46d0-b06e-89445cf255d8/astra_7ede8f23-bfd6-46d0-b06e-89445cf255d8_0a2a859731dc4205dfd0f7ceeeb6888a.jpg')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2 = cv2.resize(img2, IMAGE_SIZE)
    img2 = img2 - 127.5
    img2 = img2 * 0.0078125

    vector2 = model.predict(np.expand_dims(img2, axis=0))[0]
    print(vector2)

    dist = np.linalg.norm(vector1 - vector2)
    print(dist)


if __name__ == '__main__':
    main()
