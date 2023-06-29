
import Polygon as plg
import numpy as np
import mmcv
import os


def read_dir(root):
    file_path_list = []
    for file_path, dirs, files in os.walk(root):
        for file in files:
            file_path_list.append(os.path.join(file_path, file).replace('\\', '/'))
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
    for line in lines:
        if line == '':
            continue
        # line = util.str.remove_all(line, '\xef\xbb\xbf')
        # gt = util.str.split(line, ',')
        gt = line.split(',')

        x1 = np.int(gt[0])
        y1 = np.int(gt[1])

        bbox = [np.int(gt[i]) for i in range(4, 32)]
        bbox = np.asarray(bbox) + ([x1, y1] * 14)

        bboxes.append(bbox)
    return bboxes


def get_union(pD, pG):
    areaA = pD.area()
    areaB = pG.area()
    return areaA + areaB - get_intersection(pD, pG);


def get_intersection(pD, pG):
    pInt = pD & pG
    if len(pInt) == 0:
        return 0
    return pInt.area()


def compute_prh(pred_root, gt_root):
    th = 0.5
    pred_list = read_dir(pred_root)
    tp, fp, npos = 0, 0, 0

    for pred_path in pred_list:
        preds = get_pred(pred_path)
        gt_path = gt_root + pred_path.split('/')[-1]
        gts = get_gt(gt_path)
        npos += len(gts)

        cover = set()
        for pred_id, pred in enumerate(preds):
            pred = np.array(pred)
            pred = pred.reshape(int(pred.shape[0] / 2), 2)[:, ::-1]

            pred_p = plg.Polygon(pred)

            flag = False
            for gt_id, gt in enumerate(gts):
                gt = np.array(gt)
                gt = gt.reshape(int(gt.shape[0] / 2), 2)
                gt_p = plg.Polygon(gt)

                union = get_union(pred_p, gt_p)
                inter = get_intersection(pred_p, gt_p)

                if inter * 1.0 / union >= th:
                    if gt_id not in cover:
                        flag = True
                        cover.add(gt_id)
            if flag:
                tp += 1.0
            else:
                fp += 1.0

    # print tp, fp, npos
    precision = tp / (tp + fp)
    recall = tp / npos
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, hmean
    # print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))
