from convert_tensorflow import create_training_model

IMAGE_SIZE = (112, 112)
VERSION = 'a79'


def main():
    model = create_training_model(IMAGE_SIZE, 1, mode='infer', model_type='resnet50')

    model.load_weights('checkpoints/2021-03-05-08-30-00_e_374.ckpt')

    model.save(f'saved_model/recognition/{VERSION}/', include_optimizer=False)


if __name__ == '__main__':
    main()
