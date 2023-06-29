import torch
import torch.nn as nn

BatchNorm2d = nn.BatchNorm2d
import torch.nn.functional as F
import math
import numpy as np
import cv2
import time
from ..loss import build_loss, ohem_batch, iou
from ..utils import Conv_BN_ReLU
import copy
from PIL import Image
import os
import os.path as osp


class single_det(nn.Module):
    def __init__(self,
                 cfg,
                 in_channels,
                 hidden_dim,
                 num_classes,
                 loss_shrink,
                 loss_polar,
                 ):
        super(single_det, self).__init__()

        self.cfg = cfg

        self.header = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
            BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 2, 2),
            BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, num_classes + self.cfg.data.train.num_polars, 2, 2))

        self.shrink_loss = build_loss(loss_shrink)
        self.polar_loss = build_loss(loss_polar)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.header(f)
        return out

    def is_filter(self, shape, contour, score):
        temp = np.zeros(shape).astype(np.uint8)
        cv2.drawContours(temp, [contour], 0, 1, -1)

        index = temp > 0
        sum_mt = np.sum(temp[index])
        sum_score = np.sum(score[index])
        if ((sum_score / sum_mt) < self.cfg.test_cfg.min_score) or (sum_mt < self.cfg.test_cfg.min_area):
            return True
        else:
            return False

    def get_results(self, out, img_ori, img_meta):

        outputs = dict()

        if not self.training and self.cfg.data.test.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        # image meta
        img_path = img_meta['img_path'][0]
        ori_size = img_meta['ori_size'][0]
        img_size = img_meta['resize_size'][0]

        scale = (float(ori_size[1]) / float(img_size[1]),
                 float(ori_size[0]) / float(img_size[0]))

        ####################### 1. rebuild text contours #######################
        out0 = out[0, 0, :, :]
        pred_shrink = (out0 > 0).data.cpu().numpy().astype(np.uint8)

        score = torch.sigmoid(out0).data.cpu().numpy().astype(np.float32)
        bboxes = []

        contours, _ = cv2.findContours(pred_shrink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:

            temp, is_filter = self.is_filter(pred_shrink.shape, contour, score)
            if is_filter: continue

            area = cv2.contourArea(contour)  #### compute r through DB
            length = contour.shape[0]
            r = area * 1.5 / (length + 1e-10)

            cv2.drawContours(temp, [contour], 0, 1, int(r))

            if self.cfg.test_cfg.bbox_type == 'rect':
                points = np.array(np.where(temp)).transpose((1, 0))
                rect = cv2.minAreaRect(points[:, ::-1])
                bbox = cv2.boxPoints(rect) * scale
            elif self.cfg.test_cfg.bbox_type == 'poly':
                contours_poly, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bbox = contours_poly[0] * scale

            bbox = bbox.astype('int32')
            bboxes.append(bbox.reshape(-1))

        if not self.training and self.cfg.data.test.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                post_time=time.time() - start
            ))

        outputs.update(dict(
            bboxes=bboxes
        ))
        return outputs

    def loss(self, out, gt_shrinks, gt_instances, gt_ignores, gt_polars):
        gt_shrinks[gt_shrinks > 1] = 1
        gt_instances[gt_instances > 1] = 1
        # print(torch.unique(gt_shrinks),torch.unique(gt_instances),torch.unique(gt_ignores),torch.max(gt_polars),self.cfg.data.train.short_size)

        # 1. ------ shrink loss
        shrinks = out[:, 0, :, :]
        selected_masks = ohem_batch(shrinks, gt_instances, gt_ignores)
        loss_shrinks = self.shrink_loss(shrinks, gt_shrinks, selected_masks, reduce=False)

        # 1+. ------  shrink iou
        iou_shrinks = iou(
            (shrinks > 0), gt_shrinks, gt_instances * gt_ignores, reduce=False)

        losses = dict(
            loss_shrinks=loss_shrinks,
            iou_shrinks=iou_shrinks,
        )

        # 2. ------ polar loss
        # polars = torch.sigmoid(out[:, 1:, :, :])*self.cfg.data.train.short_size
        # gt_polars[gt_polars>self.cfg.data.train.short_size]=0
        # gt_polars = gt_polars.float()
        # loss_polars = 0
        # for i in range(polars.size()[1]):
        #     # loss_polars += self.polar_loss(polars[:,i,:,:], gt_polars[:,i,:,:], reduce=False)
        #     temploss= self.polar_loss(polars[:,i,:,:], gt_polars[:,i,:,:], reduce=False)
        #     print(loss_polars,temploss)
        #     loss_polars+=temploss
        # loss_polars = loss_polars / polars.size()[1]
        # losses.update(dict(
        #             loss_polars=loss_polars
        #         )) 

        polars = torch.sigmoid(out[:, 1:, :, :]) * self.cfg.data.train.short_size
        gt_polars[gt_polars > self.cfg.data.train.short_size] = 0

        loss_polars = self.polar_loss(polars, gt_polars, reduce=False)

        losses.update(dict(
            loss_polars=loss_polars
        ))

        return losses
