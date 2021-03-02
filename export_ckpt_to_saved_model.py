from convert_tensorflow import create_training_model

IMAGE_SIZE = (224, 224)
VERSION = 1


def main():
    model = create_training_model(IMAGE_SIZE, 1, mode='infer')

    model.load_weights('checkpoints/2020-11-18-08-51-21_e_400.ckpt')

    model.save(f'saved_model/recognition/{VERSION}/', include_optimizer=False)


if __name__ == '__main__':
    main()
