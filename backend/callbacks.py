import tensorflow as tf
import glob
import os


class MaxCkptSave(tf.keras.callbacks.Callback):
    def __init__(self, path, num_of_save):
        super().__init__()
        self.ckpt_folder_path = path
        self.num_of_save = num_of_save

    def on_epoch_end(self, epoch, logs=None):
        index_list = glob.glob(os.path.join(self.ckpt_folder_path, '*.index'))
        index_dict = {file: os.path.getctime(file) for file in index_list}
        index_dict = {k: v for k, v in sorted(index_dict.items(), key=lambda item: item[1])}
        num_of_delete = len(index_dict) - self.num_of_save
        if num_of_delete < 0:
            return
        for i, (k, _) in enumerate(index_dict.items()):
            if i == num_of_delete:
                break
            print(f'k: {k}')
            for f in glob.glob(os.path.join(f'{k[:-6]}*')):
                print(f'remove {f}')
                os.remove(f)
