from convert_tensorflow import create_training_model
import cv2
import numpy as np

num_of_class = 20000
IMAGE_SIZE = (112, 112)
CKPT = 'checkpoints/2021-04-20-10-08-21_e_5'
MODEL_TYPE = 'mobilenetv3'


def main():
    model = create_training_model(IMAGE_SIZE, num_of_class, mode='train', model_type=MODEL_TYPE)
    model.load_weights(CKPT)

    img1 = cv2.imread(
        'dataset/divide/astra_463844fe-887b-4032-9086-c976de4a24bf/astra_463844fe-887b-4032-9086-c976de4a24bf_aa0c157021ea438db44991629f35e4d0.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, IMAGE_SIZE)
    img1 = img1 - 127.5
    img1 = img1 * 0.0078125
    img1 = img1.astype(np.float32)
    img1 = np.expand_dims(img1, axis=0)

    vector1 = model([img1, np.asarray([0])]).numpy()[0]
    print(vector1)

    vector2 = model([img1, np.asarray([1])]).numpy()[0]
    print(vector2)


if __name__ == '__main__':
    main()
