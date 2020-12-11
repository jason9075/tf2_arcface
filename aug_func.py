import math
import random

import cv2
import numpy as np


def remap(img, scale_x, scale_y, center_x, center_y, radius, amount):
    h, w, _ = img.shape
    # set up the x and y maps as float32
    flex_x = np.zeros((h, w), np.float32)
    flex_y = np.zeros((h, w), np.float32)

    # create map with the barrel pincushion distortion formula
    for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
            # determine if pixel is within an ellipse
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y
            if distance >= (radius * radius):
                flex_x[y, x] = x
                flex_y[y, x] = y
            else:
                factor = 1.0
                if distance > 0.0:
                    factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
                flex_x[y, x] = factor * delta_x / scale_x + center_x
                flex_y[y, x] = factor * delta_y / scale_y + center_y

    # do the remap  this is where the magic happens
    dst = cv2.remap(img, flex_x, flex_y, cv2.INTER_LINEAR)

    return dst


def random_blur(img):
    value = random.randint(2, 4)
    value = value * 2 - 1
    img = cv2.GaussianBlur(img, (value, value), 1)

    return img


def shear_image(img):
    h, w, _ = img.shape

    M = np.float32([[1, (random.random() - 0.5) * 0.1, 0],
                    [(random.random() - 0.5) * 0.1, 1, 0],
                    [0, 0, 1]])

    img = cv2.warpPerspective(img, M, (w, h))

    return img


def slight_rotate(img):
    angle = (random.random() - 0.5) * 5
    h, w, _ = img.shape

    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h))

    return img


def add_mask(img):
    pts = [[0.05, 0.5], [0.5, 0.45], [0.95, 0.5], [0.9, 0.85], [0.5, 0.95], [0.1, 0.85]]
    pts = [[pt[0] + ((random.random() - 0.5) * 0.05), pt[1] + ((random.random() - 0.5) * 0.05)] for pt in pts]
    pts = np.array(pts) * 224
    pts = pts.astype(np.int)
    pts = pts.reshape((-1, 1, 2))

    cv2.drawContours(img, [pts], -1, (255, 230, 200), cv2.FILLED)
    return img


def distort_face(img):
    # distort mouth
    img = remap(img, 1.1, 0.5, 112, 200, random.randint(20, 25), -1)

    # distort nose
    img = remap(img, 1.1, 0.5, 112, 112, random.randint(10, 15), -1)

    # distort left face
    # img = remap(img, 0.9, 0.4, 50, 150, random.randint(35, 40), -1)

    # distort eye

    return img


def random_brightness_and_contrast(img):
    bright_value = np.random.randint(-10, 10)
    alpha_value = random.random()*0.5+0.75
    new_image = np.zeros(img.shape, img.dtype)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            for c in range(img.shape[2]):
                new_image[y, x, c] = np.clip(alpha_value * img[y, x, c] + bright_value, 0, 255)
    del img
    return new_image
