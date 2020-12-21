import csv
import glob
import os

import cv2
import tqdm

DATA_PATH = 'download'
OUTPUT_PATH = 'dataset/cele_cc_2/'


def main():
    id_folders = os.listdir(DATA_PATH)
    id_folders = [d for d in id_folders if not d.startswith('.')]

    error_list = []
    for id_name in tqdm.tqdm(id_folders):
        if not os.path.isdir(os.path.join(OUTPUT_PATH, id_name)):
            os.mkdir(os.path.join(OUTPUT_PATH, id_name))
        img_paths = glob.glob(os.path.join(DATA_PATH, id_name, '*.jpg'))
        for img_path in img_paths:
            file_name = img_path.split("/")[-1]
            if os.path.isfile(os.path.join(OUTPUT_PATH, id_name, file_name)):
                continue
            img = cv2.imread(img_path)

            if img is None:
                error_list.append(img_path)

            img = img[110:110 + 224, 88:88 + 224]
            cv2.imwrite(os.path.join(OUTPUT_PATH, id_name, file_name), img)
            del img

    print(error_list)
    with open('center_fail.csv', 'w+', newline='') as file:
        write = csv.writer(file)
        write.writerows(error_list)


if __name__ == '__main__':
    main()
