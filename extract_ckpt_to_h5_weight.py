import tensorflow as tf

from convert_tensorflow import create_training_model
from test_train import ArcFaceModel

num_of_class = 20000
IMAGE_SIZE = (112, 112)
CKPT = 'checkpoints/2021-05-04-08-06-28_e_144'
MODEL_TYPE = 'resnetv2'


def main():
    model = create_training_model(IMAGE_SIZE, num_of_class, mode='train', model_type=MODEL_TYPE)
    model.load_weights(CKPT)
    filename = CKPT.split('/')[-1]
    model.save(f'saved_model/{filename}.h5', include_optimizer=True, save_format='h5')


def test():
    model = ArcFaceModel(size=112,
                         num_classes=num_of_class,
                         embd_shape=512,
                         training=True,
                         model_type=MODEL_TYPE)
    model.summary(line_length=80)

    model.load_weights(CKPT)
    filename = CKPT.split('/')[-1]
    model.save(f'saved_model/{filename}.h5', include_optimizer=True, save_format='h5')


if __name__ == '__main__':
    # main()
    test()
