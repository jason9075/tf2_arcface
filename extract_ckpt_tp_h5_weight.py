import tensorflow as tf

from convert_tensorflow import create_training_model

num_of_class = 20000
IMAGE_SIZE = (224, 224)
CKPT = 'checkpoints/2021-02-22-09-37-54_e_3.ckpt'
FOR_TRAINING = True


def main():
    model = create_training_model(IMAGE_SIZE, 20000, mode='train')
    model.load_weights(CKPT)
    filename = CKPT.split('/')[-1]
    model.save(f'saved_model/{filename}.h5', include_optimizer=True, save_format='h5')


if __name__ == '__main__':
    main()
