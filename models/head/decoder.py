import torch
import torch.nn as nn

from ..loss.bce_loss import BalanceCrossEntropyLoss

BatchNorm2d = nn.BatchNorm2d
import torch.nn.functional as F

import math
import numpy as np
import cv2
import time
import copy
from PIL import Image
import os
import os.path as osp
from shapely.geometry import Polygon
import pyclipper
from copy import deepcopy as cdc

from ..loss import build_loss, ohem_batch, iou
from ..utils import Conv_BN_ReLU
import cc_torch
from solotext_util import clustering, ker_meaning

def points_nms(heat, kernel=2):
    # kernel must be 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=1)
    keep = (hmax[:, :, :-1, :-1] == heat).byte()
    return heat * keep


def matrix_nms(seg_masks, cate_labels, cate_scores, kernel='gaussian', sigma=2.0, sum_masks=None):
    """Matrix NMS for multi-class masks.

    Args:
        seg_masks (Tensor): shape (n, h, w)
        cate_labels (Tensor): shape (n), mask labels in descending order
        cate_scores (Tensor): shape (n), mask scores in descending order
        kernel (str):  'linear' or 'gauss'
        sigma (float): std in gaussian method
        sum_masks (Tensor): The sum of seg_masks

    Returns:
        Tensor: cate_scores_update, tensors of shape (n)
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []
    if sum_masks is None:
        sum_masks = seg_masks.sum((1, 2)).int()
    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).byte().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay
    decay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'gaussian':
        decay_matrix = torch.exp(-1 * sigma * (decay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
    elif kernel == 'linear':
        decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
        decay_coefficient, _ = decay_matrix.min(0)
    else:
        raise NotImplementedError

    # update the score.
    cate_scores_update = cate_scores * decay_coefficient
    return cate_scores_update


class Decoder(nn.Module):
    def __init__(self,
                 test_cfg,
                 in_channels,
                 hidden_dim,
                 loss_instance,
                 loss_center
                 ):
        super(Decoder, self).__init__()

        self.test_cfg = test_cfg

        # (mhz)
        self.center_back = nn.Sequential(
            Conv_BN_ReLU(in_channels, hidden_dim, 3, padding=1),
            Conv_BN_ReLU(hidden_dim, hidden_dim, 3, padding=1))

        self.center_head = nn.Conv2d(hidden_dim, 1, 3, padding=1)

        self.kernel_head = nn.Sequential(
            Conv_BN_ReLU(in_channels, hidden_dim, 3, padding=1),
            Conv_BN_ReLU(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))

        self.classification_head = nn.Sequential(
            Conv_BN_ReLU(in_channels, hidden_dim, 3, padding=1),
            Conv_BN_ReLU(hidden_dim, hidden_dim, 3, padding=1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1))

        self.pred_gt = nn.Sequential(
            Conv_BN_ReLU(in_channels, 64, 3, padding=1),
            nn.Conv2d(64, 1, 3, padding=1))

        # ConvTranspose2d 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        self.instance_loss = build_loss(loss_instance)
        self.center_loss = build_loss(loss_center)

    def forward(self, fuse):
        center_back = self.center_back(fuse)
        center = self.center_head(center_back)
        kernel = self.kernel_head(fuse)
        classification = self.classification_head(fuse)
        if not self.training:
            center = points_nms(center.sigmoid(), kernel=2)
            return [center, kernel, classification]
        return [center, kernel, classification, self.pred_gt(fuse)]

    def get_results2(self, features, cates, kernels, classifications, imgs_meta, test_cfg):
        assert len(cates) == len(kernels)
        seg_masks_all = []
        sum_masks_all = []
        cate_scores_all = []
        for i in range(len(cates)):
            cate = cates[i][0, 0, :, :]
            cate_shape = cate.shape
            cate = cate.reshape(-1)
            kernel = kernels[i][0]
            classification = classifications[i][0]
            inds = (cate > test_cfg.score_thr)
            cate_scores = cate[inds]
            if len(cate_scores) == 0:
                continue
            kernel = kernel.permute(1, 2, 0)
            kernel = kernel.reshape(-1, kernel.shape[-1])
            classification = classification.permute(1, 2, 0)
            classification = classification.reshape(-1, classification.shape[-1])
            inds = inds.nonzero()[:, 0]
            classi = classification[inds]
            fea = features[0].permute(1, 2, 0).reshape(-1, classi.shape[-1])[inds].T
            matrix = torch.matmul(classi, fea).sigmoid()
            matrix = matrix > test_cfg.matrix_thr
            matrix = matrix[torch.diag(matrix)]
            sum_column = torch.sum(matrix, dim=0)
            _, column_index = torch.sort(sum_column, descending=True)
            matrix = matrix[:, column_index]
            inds = inds[column_index]
            ker = kernel[inds]
            
            column_label = clustering(matrix.bool())
            column_unique = torch.unique(column_label)
            sum_kernel, counter, all_HW = ker_meaning(column_label, ker, column_unique, inds)
            all_mean_kernel = (sum_kernel.T/counter).T.unsqueeze(-1).unsqueeze(-1)

            seg_preds = F.conv2d(features, all_mean_kernel, stride=1).squeeze(0).sigmoid()
            seg_masks = seg_preds > test_cfg.mask_thr
            seg_preds_list = []
            seg_masks_list = []
            
            mask_connect_labels = cc_torch.connected_components_labeling(seg_masks.byte())

            H__ = all_HW // cate_shape[1]
            W__ = all_HW % cate_shape[1]
            
            all_text_labels = mask_connect_labels[:,H__,W__]
            seg_masks_len = len(seg_masks)
            for j in range(seg_masks_len):
                mask_connect_label = mask_connect_labels[j]
                # keep_unique = set(mask_connect_label[H__[j], W__[j]].cpu().numpy())
                keep_unique = set(all_text_labels[j,j].cpu().numpy())
                for k_u in keep_unique:
                    if k_u != 0:
                        mask_connect_labels_mask = mask_connect_label == k_u
                        seg_preds_list.append(j)
                        seg_masks_list.append(mask_connect_labels_mask)

            seg_preds = seg_preds[seg_preds_list]
            seg_masks = torch.stack(seg_masks_list)
            sum_masks = seg_masks.sum((1,2))
            seg_scores = (seg_preds*seg_masks).sum((1,2))/sum_masks

            keep = sum_masks > test_cfg.min_area
            if keep.sum() == 0:
                continue
            seg_masks = seg_masks[keep, ...]
            sum_masks = sum_masks[keep, ...]
            seg_scores = seg_scores[keep, ...]
            seg_masks_all.append(seg_masks)
            sum_masks_all.append(sum_masks)
            cate_scores_all.append(seg_scores)
            
        if len(seg_masks_all) == 0:
            return None

        seg_masks = torch.cat(seg_masks_all)
        sum_masks = torch.cat(sum_masks_all)
        cate_scores = torch.cat(cate_scores_all)

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > test_cfg.nms_pre:
            sort_inds = sort_inds[:test_cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]

        cate_labels = torch.ones_like(cate_scores).byte()
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, test_cfg.kernel, test_cfg.sigma, sum_masks)

        # filter.
        keep = cate_scores >= self.test_cfg.min_score
        if keep.sum() == 0:
            return None

        seg_masks = seg_masks[keep, :, :]
        return seg_masks

    def get_HW2(self, features, cates, kernels, imgs_meta, test_cfg):
        assert len(cates) == len(kernels)
        H_ratio = (imgs_meta['ori_size'] / imgs_meta['resize_size'])[0, 0]
        W_ratio = (imgs_meta['ori_size'] / imgs_meta['resize_size'])[0, 1]

        contours_count = []
        contours_contain = []

        cate = cates[0][0, 0, :, :]
        cate_shape = cate.shape
        cate = cate.reshape(-1)
        kernel = kernels[0][0]
        inds = (cate > test_cfg.score_thr)
        cate_scores = cate[inds]
        if len(cate_scores) == 0:
            return [], [], [], [], None
        kernel = kernel.permute(1, 2, 0)
        kernel = kernel.reshape(-1, kernel.shape[-1])
        inds = inds.nonzero()[:, 0]
        ker = kernel[inds]
        # 下面利用ker对feature进行卷积
        seg_preds = F.conv2d(features, ker.unsqueeze(-1).unsqueeze(-1), stride=1).squeeze(0).sigmoid()
        seg_masks = seg_preds > test_cfg.mask_thr

        H = (inds // cate_shape[1])
        W = (inds % cate_shape[1])
        for j in range(seg_masks.shape[0]):
            mask_connect_labels = cc_torch.connected_components_labeling(seg_masks[j].byte())
            contours_count.append(len(torch.unique(mask_connect_labels)) - 1)
            h, w = int(H[j]), int(W[j])
            contours_contain.append(int(int(mask_connect_labels[h, w]) != 0))
            seg_masks[j][mask_connect_labels != int(mask_connect_labels[h, w])] = 0

        H = H * 4 * H_ratio
        W = W * 4 * W_ratio
        seg_masks = F.interpolate(seg_masks.unsqueeze(0).byte(), size=imgs_meta['resize_size'].numpy()[0].tolist(), mode='nearest')
        seg_masks = F.interpolate(seg_masks, size=imgs_meta['ori_size'].numpy()[0].tolist(), mode='nearest').squeeze(0)

        return H, W, contours_count, contours_contain, seg_masks

    def get_HW2_nms(self, features, cates, kernels, imgs_meta, test_cfg):
        assert len(cates) == len(kernels)
        H_ratio = (imgs_meta['ori_size'] / imgs_meta['resize_size'])[0, 0]
        W_ratio = (imgs_meta['ori_size'] / imgs_meta['resize_size'])[0, 1]

        seg_masks_all = []
        sum_masks_all = []
        cate_scores_all = []

        contours_count = []
        contours_contain = []

        cate = cates[0][0, 0, :, :]
        cate_shape = cate.shape
        cate = cate.reshape(-1)
        kernel = kernels[0][0]
        inds = (cate > test_cfg.score_thr)
        cate_scores = cate[inds]
        if len(cate_scores) == 0:
            return [], [], [], [], None
        kernel = kernel.permute(1, 2, 0)
        kernel = kernel.reshape(-1, kernel.shape[-1])
        inds = inds.nonzero()[:, 0]
        ker = kernel[inds]
        # 下面利用ker对feature进行卷积
        seg_preds = F.conv2d(features, ker.unsqueeze(-1).unsqueeze(-1), stride=1).squeeze(0).sigmoid()
        seg_masks = seg_preds > test_cfg.mask_thr

        H = (inds // cate_shape[1])
        W = (inds % cate_shape[1])
        for j in range(seg_masks.shape[0]):
            mask_connect_labels = cc_torch.connected_components_labeling(seg_masks[j].byte())
            contours_count.append(len(torch.unique(mask_connect_labels)) - 1)
            h, w = int(H[j]), int(W[j])
            contours_contain.append(int(int(mask_connect_labels[h, w]) != 0))
            seg_masks[j][mask_connect_labels != int(mask_connect_labels[h, w])] = 0

        sum_masks = seg_masks.sum((1, 2)).int()

        keep = sum_masks > test_cfg.min_area
        if keep.sum() == 0:
            return [], [], [], [], None

        seg_preds = seg_preds[keep, ...]
        seg_masks = seg_masks[keep, ...]
        sum_masks = sum_masks[keep]

        contours_contain = np.array(contours_contain)[keep.cpu().numpy()]
        contours_count = np.array(contours_count)[keep.cpu().numpy()]
        H = H[keep]
        W = W[keep]
        # maskness.
        seg_scores = (seg_preds * seg_masks.byte()).sum((1, 2)) / sum_masks

        seg_masks_all.append(seg_masks)
        sum_masks_all.append(sum_masks)
        cate_scores_all.append(seg_scores)

        seg_masks = torch.cat(seg_masks_all)
        sum_masks = torch.cat(sum_masks_all)
        cate_scores = torch.cat(cate_scores_all)

        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > test_cfg.nms_pre:
            sort_inds = sort_inds[:test_cfg.nms_pre]
        seg_masks = seg_masks[sort_inds, :, :]
        sum_masks = sum_masks[sort_inds]
        cate_scores = cate_scores[sort_inds]
        H = H[sort_inds]
        W = W[sort_inds]
        contours_count = contours_count[sort_inds.cpu().numpy()]
        contours_contain = contours_contain[sort_inds.cpu().numpy()]

        cate_labels = torch.ones_like(cate_scores).byte()
        cate_scores = matrix_nms(seg_masks, cate_labels, cate_scores, test_cfg.kernel, test_cfg.sigma, sum_masks)

        # filter.
        keep = cate_scores >= self.test_cfg.min_score
        if keep.sum() == 0:
            return [], [], [], [], None

        seg_masks = seg_masks[keep, :, :]
        cate_scores = cate_scores[keep]
        H = H[keep]
        W = W[keep]
        contours_contain = contours_contain[keep.cpu().numpy()]
        contours_count = contours_count[keep.cpu().numpy()]

        # sort and keep top_k
        sort_inds = torch.argsort(cate_scores, descending=True)
        if len(sort_inds) > test_cfg.max_per_img:
            sort_inds = sort_inds[:test_cfg.max_per_img]
        seg_masks = seg_masks[sort_inds, :, :]
        H = H[sort_inds]
        W = W[sort_inds]
        contours_contain = contours_contain[sort_inds.cpu().numpy()]
        contours_count = contours_count[sort_inds.cpu().numpy()]

        H = H * 4 * H_ratio
        W = W * 4 * W_ratio
        seg_masks = F.interpolate(seg_masks.unsqueeze(0).byte(), size=imgs_meta['resize_size'].numpy()[0].tolist(), mode='nearest')
        seg_masks = F.interpolate(seg_masks, size=imgs_meta['ori_size'].numpy()[0].tolist(), mode='nearest').squeeze(0)

        if len(seg_masks) == 1:
            return H, W, [contours_count], [contours_contain], seg_masks

        return H, W, contours_count, contours_contain, seg_masks

    def get_all_mask(self, features, cates, kernels, imgs_meta, test_cfg):
        assert len(cates) == len(kernels)
        H_ratio = (imgs_meta['ori_size'] / imgs_meta['resize_size'])[0, 0]
        W_ratio = (imgs_meta['ori_size'] / imgs_meta['resize_size'])[0, 1]

        cate = cates[0][0, 0, :, :]
        cate_shape = cate.shape
        cate = cate.reshape(-1)
        kernel = kernels[0][0]
        inds = (cate > test_cfg.score_thr)
        cate_scores = cate[inds]
        if len(cate_scores) == 0:
            return [], [], None
        kernel = kernel.permute(1, 2, 0)
        kernel = kernel.reshape(-1, kernel.shape[-1])
        inds = inds.nonzero()[:, 0]
        ker = kernel[inds]
        # 下面利用ker对feature进行卷积
        seg_preds = F.conv2d(features, ker.unsqueeze(-1).unsqueeze(-1), stride=1).squeeze(0).sigmoid()
        seg_masks = seg_preds > test_cfg.mask_thr

        H = (inds // cate_shape[1])
        W = (inds % cate_shape[1])
        H = H * 4 * H_ratio
        W = W * 4 * W_ratio
        seg_masks = F.interpolate(seg_masks.unsqueeze(0).byte(), size=imgs_meta['resize_size'].numpy()[0].tolist(), mode='nearest')
        seg_masks = F.interpolate(seg_masks, size=imgs_meta['ori_size'].numpy()[0].tolist(), mode='nearest').squeeze(0)

        return H, W, seg_masks

    def loss(self,
             features,
             pred_gt,
             cates,
             kernels,
             classifications,
             gt_instance,
             gt_ignore,
             gt_center,
             gt_centerness,
             gt_dminimum):

        batch = features.shape[0]
        count = len(cates)

        loss_mask = 0
        mask_count = 0
        ignore = F.interpolate(gt_ignore.unsqueeze(0).float(), size=gt_instance[0].shape)[0]
        for i in range(count):
            kernel = kernels[i]
            center_gt = gt_center[count - 1 - i]
            for j in range(batch):
                ker = kernel[j]
                per_img = center_gt[j]
                for u in torch.unique(per_img):
                    if u == 0:
                        continue
                    k = ker.permute(1, 2, 0)[per_img == u]
                    ins_gt = (gt_instance[j] == u).int()
                    for index in range(k.shape[0]):
                        ins_pred = F.conv2d(features[j].unsqueeze(0), k[index].reshape(1, -1, 1, 1), stride=1)[0][0]
                        loss_mask += self.instance_loss(ins_pred, ins_gt, ignore[j])
                        mask_count += 1
        losses = dict(loss_mask=loss_mask / mask_count)


        loss_fuse_gt = 0
        for j in range(batch):
            pred_gt_single = pred_gt[j, 0]
            gt_single = (gt_instance[j]!=0).int()
            loss_fuse_gt+=self.instance_loss(pred_gt_single, gt_single, ignore[j])
        losses.update(loss_fuse_gt=loss_fuse_gt / batch)


        loss_classification = 0
        classification_count = 0
        for i in range(count):
            classification = classifications[i]
            center_gt = gt_center[count - 1 - i]
            for j in range(batch):
                classification_list = []
                feature_list = []
                per_img = center_gt[j]
                center_count = torch.sum(per_img != 0)
                target_m = torch.zeros(center_count, center_count).cuda()
                start = 0
                for u in torch.unique(per_img):
                    if u == 0:
                        continue
                    classification_list.append(classification[j].permute(1, 2, 0)[per_img == u])
                    feature_list.append(features[j].permute(1, 2, 0)[per_img == u])
                    temp_count = torch.sum(per_img == u)
                    target_m[start:start + temp_count, start:start + temp_count] = 1
                    start += temp_count
                if len(feature_list) != 0:
                    pred_m = torch.mm(torch.cat(classification_list), torch.cat(feature_list).T)
                    idx = torch.randperm(target_m.nelement())
                    target_m = target_m.view(-1)[idx].view(target_m.size())
                    pred_m = pred_m.view(-1)[idx].view(pred_m.size())
                    loss_classification += self.center_loss(pred_m, target_m, reduction_override='sum')
                    classification_count += int(torch.sum(target_m))
        losses.update(loss_classification=loss_classification / (classification_count + 1))

        loss_cates = 0
        all_ins = 0
        for i in range(count):
            center_pred = cates[i][:, 0, :, :]
            center_gt = gt_center[count - 1 - i]
            center_gt_temp = (center_gt > 0).int()
            for j in range(batch):
                loss_cates += self.center_loss(center_pred[j], center_gt_temp[j], weight=ignore[j], reduction_override='sum')
                all_ins += int(torch.sum(center_gt_temp[j]))
        losses.update(loss_cates=loss_cates / (all_ins + 1))

        return losses
