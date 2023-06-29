import numpy as np
import cv2
import random
import math


#### test_augmentation
def scale_aligned_short(img, short_size):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


#### train_augmentation
def scale_aligned(img, scale):
    h, w = img.shape[0:2]
    h = int(h * scale + 0.5)
    w = int(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


def random_scale(datas, short_size):
    h, w = datas[0].shape[0:2]

    random_scale = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    scale = (np.random.choice(random_scale) * short_size) / min(h, w)

    # datas: [img, ignore, instance]
    for i in range(len(datas)):
        if i < 2:
            datas[i] = scale_aligned(datas[i], scale)
        else:
            uniques = np.unique(datas[i])[1:]
            data = np.zeros(datas[0].shape[:2])
            for idx in uniques:
                dataIdx = (datas[i] == idx).astype(np.uint8)
                dataIdx = scale_aligned(dataIdx, scale)
                data += (dataIdx * idx)
                data[data > idx] = idx
            datas[i] = data
    return datas


def random_horizontal_flip(datas):
    if random.random() < 0.5:
        for i in range(len(datas)):
            datas[i] = np.flip(datas[i], axis=1).copy()
    return datas


def random_rotate(datas):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle

    h, w = datas[0].shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    # datas: [img, ignore, instance]
    for i in range(len(datas)):
        if i < 2:
            datas[i] = cv2.warpAffine(datas[i], rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
        else:
            uniques = np.unique(datas[i])[1:]
            data = np.zeros(datas[0].shape[:2])
            for idx in uniques:
                dataIdx = (datas[i] == idx).astype(np.uint8)
                dataIdx = cv2.warpAffine(dataIdx, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
                data += (dataIdx * idx)
                data[data > idx] = 0
            datas[i] = data
    return datas


def random_crop_padding(datas, target_size):
    h, w = datas[0].shape[0:2]

    t_w, t_h = target_size, target_size
    p_w, p_h = target_size, target_size
    if w == t_w and h == t_h:
        return datas

    t_h = t_h if t_h < h else h
    t_w = t_w if t_w < w else w

    if random.random() > 3.0 / 8.0 and np.max(datas[1]) > 0:
        # make sure to crop the text region
        tl = np.min(np.where(datas[1] > 0), axis=1) - (t_h, t_w)
        tl[tl < 0] = 0
        br = np.max(np.where(datas[1] > 0), axis=1) - (t_h, t_w)
        br[br < 0] = 0
        br[0] = min(br[0], h - t_h)
        br[1] = min(br[1], w - t_w)

        i = random.randint(tl[0], br[0]) if tl[0] < br[0] else 0
        j = random.randint(tl[1], br[1]) if tl[1] < br[1] else 0
    else:
        i = random.randint(0, h - t_h) if h - t_h > 0 else 0
        j = random.randint(0, w - t_w) if w - t_w > 0 else 0

    # datas: [img, ignore, instance]
    new_datas = []
    for idx in range(len(datas)):
        if idx == 0:
            s3_length = int(datas[idx].shape[-1])
            dataIdx = datas[idx][i:i + t_h, j:j + t_w, :]
            dataCopy = cv2.copyMakeBorder(dataIdx, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                          value=tuple(0 for i in range(s3_length)))
        else:
            dataIdx = datas[idx][i:i + t_h, j:j + t_w]
            dataCopy = cv2.copyMakeBorder(dataIdx, 0, p_h - t_h, 0, p_w - t_w, borderType=cv2.BORDER_CONSTANT,
                                          value=(0,))
        new_datas.append(dataCopy)
    return new_datas
