import cv2
import numpy as np
from skimage import transform as trans

SIZE = 224
left_profile_landmarks = np.array([
    [0.35, 0.25],
    [0.62, 0.21],
    [0.32, 0.50],
    [0.36, 0.76],
    [0.60, 0.76]], dtype=np.float32)
left_profile_landmarks *= SIZE

left_landmarks = np.array([
    [0.29, 0.25],
    [0.69, 0.21],
    [0.35, 0.50],
    [0.30, 0.73],
    [0.67, 0.73]], dtype=np.float32)
left_landmarks *= SIZE

front_landmarks = np.array([
    [0.24, 0.25],
    [0.76, 0.25],
    [0.50, 0.53],
    [0.28, 0.74],
    [0.72, 0.74]], dtype=np.float32)
front_landmarks *= SIZE

right_landmarks = np.array([
    [0.31, 0.21],
    [0.71, 0.25],
    [0.65, 0.50],
    [0.33, 0.73],
    [0.70, 0.73]], dtype=np.float32)
right_landmarks *= SIZE

right_profile_landmarks = np.array([
    [0.38, 0.21],
    [0.65, 0.25],
    [0.68, 0.50],
    [0.40, 0.76],
    [0.64, 0.76]], dtype=np.float32)
right_profile_landmarks *= SIZE

landmark_src = np.array([left_profile_landmarks, left_landmarks,
                         front_landmarks, right_landmarks, right_profile_landmarks])


def estimate_norm(lmk):
    assert lmk.shape == (5, 2)
    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(lmk, 2, values=np.ones(5), axis=1)
    min_M = []
    min_index = []
    min_error = float('inf')
    for i in range(5):
        tform.estimate(lmk, landmark_src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - landmark_src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_M = M
            min_index = i
    return min_M, min_index


def align(img, landmark, image_size):
    M, pose_index = estimate_norm(landmark)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped
