import Polygon as plg
import numpy as np
import mmcv
import os
import math
import cv2


def read_dir(root):
    file_path_list = []
    for file_path, dirs, files in os.walk(root):
        for file in files:
            file_path_list.append(os.path.join(file_path, file).replace('\\', '/'))
    file_path_list.sort()
    return file_path_list


def read_dir_mhz(root):
    file_path_list = []
    for file in os.listdir(root):
        path = os.path.join(root, file).replace('\\', '/')
        if os.path.isfile(path):
            file_path_list.append(path)
    file_path_list.sort()
    return file_path_list


def read_file(file_path):
    file_object = open(file_path, 'r')
    file_content = file_object.read()
    file_object.close()
    return file_content


def write_file(file_path, file_content):
    if file_path.find('/') != -1:
        father_dir = '/'.join(file_path.split('/')[0:-1])
        if not os.path.exists(father_dir):
            os.makedirs(father_dir)
    file_object = open(file_path, 'w')
    file_object.write(file_content)
    file_object.close()


def write_file_not_cover(file_path, file_content):
    father_dir = '/'.join(file_path.split('/')[0:-1])
    if not os.path.exists(father_dir):
        os.makedirs(father_dir)
    file_object = open(file_path, 'a')
    file_object.write(file_content)
    file_object.close()


def get_pred(path):
    lines = read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        if len(bbox) % 2 == 1:
            print(path)
        bbox = [int(x) for x in bbox]
        bboxes.append(bbox)
    return bboxes


def get_gt(path):
    lines = read_file(path).split('\n')
    bboxes = []
    tags = []
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ' ')
        gt = line.split(' ')

        w_ = np.float(gt[4])
        h_ = np.float(gt[5])
        x1 = np.float(gt[2]) + w_ / 2.0
        y1 = np.float(gt[3]) + h_ / 2.0
        theta = np.float(gt[6]) / math.pi * 180

        bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
        bbox = bbox.reshape(-1)

        bboxes.append(bbox)
        tags.append(np.int(gt[1]))
    return np.array(bboxes), tags


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG)


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def compute_prh(pred_root, gt_root):
    th = 0.5
    pred_list = read_dir_mhz(pred_root)

    count, tp, fp, tn, ta = 0, 0, 0, 0, 0
    for pred_path in pred_list:
        count = count + 1
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1].split('.')[0] + '.gt'
        gts, tags = get_gt(gt_path)

        ta = ta + len(preds)
        for gt, tag in zip(gts, tags):
            gt = np.array(gt)
            gt = gt.reshape(int(gt.shape[0] / 2), 2)
            gt_p = plg.Polygon(gt)
            difficult = tag
            flag = 0
            for pred in preds:
                pred = np.array(pred)
                pred = pred.reshape(int(pred.shape[0] / 2), 2)
                pred_p = plg.Polygon(pred)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)
                iou = float(inter) / union
                if iou >= th:
                    flag = 1
                    tp = tp + 1
                    break

            if flag == 0 and difficult == 0:
                fp = fp + 1

    recall = float(tp) / (tp + fp)
    if ta == 0:
        precision = 0
    else:
        precision = float(tp) / ta
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, hmean
