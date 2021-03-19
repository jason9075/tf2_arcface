from convert_tensorflow import create_training_model

IMAGE_SIZE = (112, 112)
VERSION = 'aws'


def main():
    model = create_training_model(IMAGE_SIZE, 1, mode='infer', model_type='mobilenetv2')

    model.load_weights('checkpoints/2021-03-02-09-50-25_e_10')

    model.save(f'saved_model/recognition/{VERSION}/', include_optimizer=False)


if __name__ == '__main__':
    main()
