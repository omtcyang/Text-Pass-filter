import sys
import traceback
import numpy as np
import cv2
from .util import *


#### generating train label
def supervision_label(ins_resize,
                      unique,
                      sample_stride,
                      sample_type,
                      contour,
                      contour_resize,
                      sample_scale_range,
                      center,
                      centerness,
                      dminimum):
    # 计算中心点坐标
    coords = np.array(np.where(ins_resize == 1)).transpose((1, 0))[:, ::-1]

    xmin, xmax, ymin, ymax = np.min(coords[:, 0]), np.max(coords[:, 0]), np.min(coords[:, 1]), np.max(coords[:, 1])

    dx, dy = xmax - xmin, ymax - ymin
    if dx >= dy:
        x = int(xmin + dx / 2)
        y_pts = coords[coords[:, 0] == x][:, 1]
        if y_pts.shape[0] > 0:
            y_pts = np.sort(y_pts)
            y = y_pts[len(y_pts) // 2]
            centerPt_resize = (x, y)
    else:
        y = int(ymin + dy / 2)
        x_pts = coords[coords[:, 1] == y][:, 0]
        if x_pts.shape[0] > 0:
            x_pts = np.sort(x_pts)
            x = x_pts[len(x_pts) // 2]
            centerPt_resize = (x, y)

    if sample_type == 'multi':
        # 在中心轴上采样-------------------------------------------------------------------------------------
        ds_temp = np.sqrt((contour[:, 0] - centerPt_resize[0] * sample_stride) ** 2
                          + (contour[:, 1] - centerPt_resize[1] * sample_stride) ** 2)
        if np.min(ds_temp) <= sample_scale_range[0] or np.min(ds_temp) > sample_scale_range[1]:
            return

        x_list = []
        y_list = []
        count = 5
        if dx >= dy:
            start = xmin + 0.2 * dx
            end = xmin + 0.8 * dx
            step = (end - start) / (count - 1)
            xlist = set([int(start + i * step) for i in range(0, count)])
            for x_temp in xlist:
                y_pts = coords[coords[:, 0] == x_temp][:, 1]
                if y_pts.shape[0] > 0:
                    y_pts = np.sort(y_pts)
                    y_temp = y_pts[len(y_pts) // 2]
                    x_list.append(x_temp)
                    y_list.append(y_temp)
        else:
            start = ymin + 0.2 * dy
            end = ymin + 0.8 * dy
            step = (end - start) / (count - 1)
            ylist = set([int(start + i * step) for i in range(0, count)])
            for y_temp in ylist:
                x_pts = coords[coords[:, 1] == y_temp][:, 0]
                if x_pts.shape[0] > 0:
                    x_pts = np.sort(x_pts)
                    x_temp = x_pts[len(x_pts) // 2]
                    x_list.append(x_temp)
                    y_list.append(y_temp)

        for index in range(len(y_list)):
            coorY = y_list[index]
            coorX = x_list[index]
            #### center
            center[coorY, coorX] = unique
            #### dminimum
            ds_temp = np.sqrt(
                (contour[:, 0] - coorX * sample_stride) ** 2 + (contour[:, 1] - coorY * sample_stride) ** 2)
            dminimum[coorY, coorX] = np.min(ds_temp)
            ####· centerness
            d_temp = np.sqrt((centerPt_resize[0] * sample_stride - coorX * sample_stride) ** 2
                             + (centerPt_resize[1] * sample_stride - coorY * sample_stride) ** 2)
            centerness[coorY, coorX] = np.min(ds_temp) / (np.min(ds_temp) + d_temp + 1e-6)

    if sample_type == 'single':
        ds_temp = np.sqrt((contour[:, 0] - centerPt_resize[0] * sample_stride) ** 2
                          + (contour[:, 1] - centerPt_resize[1] * sample_stride) ** 2)

        if np.min(ds_temp) <= sample_scale_range[0] or np.min(ds_temp) > sample_scale_range[1]:
            return
        dminimum[centerPt_resize[1], centerPt_resize[0]] = np.min(ds_temp)
        ####· centerness
        centerness[centerPt_resize[1], centerPt_resize[0]] = 1
        #### center
        center[centerPt_resize[1], centerPt_resize[0]] = unique


def supervision_generate(datas, img_path, sample_stride, sample_type, sample_scale_ranges, mask_size):
    img, ignore, instance = datas[0], datas[1], datas[2]
    h, w = instance.shape

    centers, centernesses, dminimums = [], [], []
    uniques = np.unique(instance)[1:]
    origin_contours = []
    for unique in uniques:
        ins = (instance == unique).astype(np.uint8)
        contours, _ = cv2.findContours(ins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 1:
            for contour in contours:
                cv2.drawContours(ignore, [contour], 0, 0, -1)
        else:
            origin_contours.append(contours)

    instance = instance * ignore

    uniques = np.unique(instance)[1:]
    for sl_index, sl in enumerate(sample_stride):
        center = np.zeros((h // sl, w // sl)).astype(np.uint8)
        centerness = np.zeros((h // sl, w // sl)).astype(np.float32)
        dminimum = np.zeros((h // sl, w // sl)).astype(np.float32)

        for unique, origin_contour in zip(uniques, origin_contours):
            ins = (instance == unique).astype(np.uint8)
            ins = cv2.resize(ins, dsize=(w // sl, h // sl))
            contours, _ = cv2.findContours(ins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 1:
                # 这里不应该对ignore进行重新赋值，因此去除
                # for co in origin_contour:
                #     cv2.drawContours(ignore, [co], 0, 0, -1)
                continue

            supervision_label(ins, unique, sl, sample_type, origin_contour[0][:, 0, :], contours[0][:, 0, :],
                              sample_scale_ranges[sl_index], center, centerness, dminimum)

        centers.append(center)
        centernesses.append(centerness)
        dminimums.append(dminimum)

    datas[1], datas[2] = ignore, instance * ignore
    datas.append(centers)
    datas.append(centernesses)
    datas.append(dminimums)
    return datas
