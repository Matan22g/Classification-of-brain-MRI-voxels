
"""By Matan Achiel 205642119, Netanel Moyal 307888974"""

import os
import numpy as np
import cv2

TRAINING_DIR = r'training'
VALIDATION_DIR = r'validation'


def train_augmention(image, im_path):
    im_aug_lst = []
    im_class_lst = []
    im_aug_lst.append(np.flipud(image).flatten())
    im_aug_lst.append(np.fliplr(image).flatten())
    for i in range(3):
        im_aug_lst.append(np.rot90(image, i + 1).flatten())
    for i in range(5):
        if "neg" in im_path:
            im_class_lst.append(0)
        else:
            im_class_lst.append(1)
    return im_aug_lst, im_class_lst


def load_dataset(augmention=True, train_dir=TRAINING_DIR, test_dir=VALIDATION_DIR):
    img_data = []
    label_data = []
    for dir in [train_dir, test_dir]:
        img_data_array = []
        class_name = []
        for file in os.listdir(dir):
            image_path = os.path.join(dir, file)
            image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
            image = np.array(image)
            image = image.astype('float32')
            image /= 255

            if dir == train_dir:
                if augmention:
                    aug_pics, aug_classes = train_augmention(image, image_path)
                    img_data_array += aug_pics
                    class_name += aug_classes

            image = image.flatten()

            img_data_array.append(image)
            if "neg" in image_path:
                class_name.append(0)
            else:
                class_name.append(1)

        img_data.append(img_data_array)
        label_data.append(class_name)

    return [img_data[0], label_data[0]], [img_data[1], label_data[1]]

# extract the image array and class name
# train_data, test_data = create_dataset(r'training')
# print(train_data, test_data)
