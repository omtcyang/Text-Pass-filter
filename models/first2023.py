import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import time

from .backbone import build_backbone
from .neck import build_neck, SPPF
from .head import build_head
from .utils import Conv_BN_ReLU
from copy import deepcopy as cdc
from dotted.collection import DottedDict
from .head.feature_head2 import FeatureHead2
import os
import cv2
# from sklearn.decomposition import PCA


class First2023(nn.Module):
    def __init__(self,
                 backbone,
                 neck,
                 head,
                 featurehead,
                 count):
        super(First2023, self).__init__()

        self.backbone = build_backbone(backbone)
        self.type = count

        neck_t = cdc(neck.__dict__['store'])
        neck_t['type'] = neck_t['type'] + self.type
        self.neck = build_neck(DottedDict(neck_t))
        self.head = build_head(head)
        self.feature_head = build_head(featurehead)

        self.report_speed = head.test_cfg.report_speed
        self.test_cfg = head.test_cfg

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.upsample(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self,
                imgs,
                gt_instance=None,
                gt_ignore=None,
                gt_center=None,
                gt_centerness=None,
                gt_dminimum=None,
                imgs_ori=None,
                imgs_meta=None):
        outputs = dict()

        #### backbone
        if not self.training and self.report_speed:
            torch.cuda.synchronize()
            start = time.time()

        f = self.backbone(imgs)

        if not self.training and self.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                backbone_time=time.time() - start
            ))
            start = time.time()

        #### FPN
        fuse = self.neck(f[0], f[1], f[2], f[3])
        # H,W = fuse.shape[-2:]
        # pca = PCA(1)
        # fuse_img = pca.fit_transform(fuse[0].permute(1,2,0).reshape(H*W,-1).cpu().numpy()).reshape(H,W)
        # cv2.imwrite('fpn/{}'.format(imgs_meta['img_path'][0].split("/")[-1]),fuse_img)


        # 下面对fuse进行降维可视化
        # H,W = fuse.shape[-2:]
        # pca = PCA(3)
        # fuse_img = pca.fit_transform(fuse[0].permute(1,2,0).reshape(H*W,-1).cpu().numpy()).reshape(H,W,3)
        # cv2.imwrite('img/{}'.format(imgs_meta['img_path'][0].split("/")[-1]),fuse_img)
        # ------------------------------------------







        if not self.training and self.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                neck_time=time.time() - start
            ))
            start = time.time()

        #### head
        cates = []
        kernels = []
        classifications = []

        if self.type == '1':
            if self.training:
                cate, kernel, classification, pred_gt = self.head(fuse)
            else:
                cate, kernel, classification = self.head(fuse)
            # if self.training:
            #     features, ins = self.feature_head(fuse)
            # else:
            features = self.feature_head(fuse)
            


            # H,W = cate.shape[-2:]
            # pca = PCA(3)
            # fuse_img = pca.fit_transform(cate[0].permute(1,2,0).reshape(H*W,-1).cpu().numpy()).reshape(H,W,3)
            # cv2.imwrite('img/{}'.format(imgs_meta['img_path'][0].split("/")[-1]),fuse_img)


            cates.append(cate)
            kernels.append(kernel)
            classifications.append(classification)
        else:
            if self.type == '4':
                x_r = torch.linspace(-1, 1, fuse[0].shape[-1], device=fuse[0].device)
                y_r = torch.linspace(-1, 1, fuse[0].shape[-2], device=fuse[0].device)
                y, x = torch.meshgrid(y_r, x_r)
                y = y.expand([fuse[0].shape[0], 1, -1, -1])
                x = x.expand([fuse[0].shape[0], 1, -1, -1])
                xy = F.upsample(torch.cat([x, y], 1), scale_factor=8, mode='bilinear')
                a = F.upsample(fuse[0], scale_factor=8, mode='nearest')
                b = F.upsample(fuse[1], scale_factor=4, mode='nearest')
                c = F.upsample(fuse[2], scale_factor=2, mode='nearest')
                d = fuse[3]
                featureConcat = torch.cat((xy, a, b, c, d), 1)
                # featureConcat = fuse[::-1]
            else:
                a = F.upsample(fuse[0], scale_factor=4, mode='nearest')
                b = fuse[1]
                featureConcat = torch.cat((a, b), 1)
            features = self.feature_head(featureConcat)
            for f in fuse:
                cate, kernel = self.head(f)
                cates.append(cate)
                kernels.append(kernel)

            # cate, kernel = self.head(torch.cat([a, b, c, d], 1))
            # cates.append(cate)
            # kernels.append(kernel)

        # print('>>>>>> head',head_out[0].shape,head_out[1].shape,head_out[2].shape)
        if not self.training and self.report_speed:
            torch.cuda.synchronize()
            outputs.update(dict(
                head_time=time.time() - start
            ))
            start = time.time()

        #### loss and post-processing
        if self.training:
            det_loss = self.head.loss(
                features,
                pred_gt,
                cates,
                kernels,
                classifications,
                gt_instance,
                gt_ignore,
                gt_center,
                gt_centerness,
                gt_dminimum)
            outputs.update(det_loss)
        else:
            seg_masks = self.head.get_results2(features, cates, kernels, classifications, imgs_meta, self.test_cfg)
            if not self.training and self.report_speed:
                torch.cuda.synchronize()
                outputs.update(dict(
                    post_time=time.time() - start
                ))
            # print(seg_masks.shape)
            
            bboxes = []
            # res_masks = np.zeros(imgs_meta['ori_size'].numpy()[0].tolist())
            # origin_img = cv2.imread(imgs_meta['img_path'][0])
            if seg_masks is not None:
                seg_masks = F.interpolate(seg_masks.unsqueeze(0).byte(), size=imgs_meta['resize_size'].numpy()[0].tolist(), mode='nearest')
                seg_masks = F.interpolate(seg_masks, size=imgs_meta['ori_size'].numpy()[0].tolist(), mode='nearest').squeeze(0)
                for instance in seg_masks:
                    ins = instance.cpu().numpy().astype(np.uint8)
                    # res_masks+=ins
                    contours_poly, _ = cv2.findContours(ins, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # cv2.drawContours(origin_img,[contours_poly[0]],-1,(0,255,0),2)
                    if self.test_cfg.bbox_type == 'poly':
                        bbox = contours_poly[0].astype('int32')
                    else:
                        rect = cv2.minAreaRect(contours_poly[0])
                        bbox = cv2.boxPoints(rect).astype('int32')
                    bboxes.append(bbox.reshape(-1))
            outputs.update({'bboxes': bboxes})
            # cv2.imwrite('mask/{}'.format(imgs_meta['img_path'][0].split("/")[-1]),origin_img)
        
        return outputs

    def getHW(self,
              imgs,
              gt_instance=None,
              gt_ignore=None,
              gt_center=None,
              gt_centerness=None,
              gt_dminimum=None,
              imgs_ori=None,
              imgs_meta=None):
        f = self.backbone(imgs)
        #### FPN
        fuse = self.neck(f[0], f[1], f[2], f[3])

        #### head
        cates = []
        kernels = []

        if self.type == '1':
            cate, kernel = self.head(fuse)
            features = self.feature_head(fuse)
            cates.append(cate)
            kernels.append(kernel)
        else:
            if self.type == '4':
                a = F.upsample(fuse[0], scale_factor=8, mode='nearest')
                b = F.upsample(fuse[1], scale_factor=4, mode='nearest')
                c = F.upsample(fuse[2], scale_factor=2, mode='nearest')
                d = fuse[3]
                featureConcat = torch.cat((a, b, c, d), 1)
            else:
                a = F.upsample(fuse[0], scale_factor=4, mode='nearest')
                b = fuse[1]
                featureConcat = torch.cat((a, b), 1)
            features = self.feature_head(featureConcat)
            for f in fuse:
                cate, kernel = self.head(f)
                cates.append(cate)
                kernels.append(kernel)

        HW, contours_count, contains = self.head.get_HW(features, cates, kernels, imgs_meta, self.test_cfg)
        return HW, contours_count, contains

    def getHW2(self,
               imgs,
               gt_instance=None,
               gt_ignore=None,
               gt_center=None,
               gt_centerness=None,
               gt_dminimum=None,
               imgs_ori=None,
               imgs_meta=None):
        f = self.backbone(imgs)
        #### FPN
        fuse = self.neck(f[0], f[1], f[2], f[3])

        #### head
        cates = []
        kernels = []

        if self.type == '1':
            cate, kernel = self.head(fuse)
            features = self.feature_head(fuse)
            cates.append(cate)
            kernels.append(kernel)
        else:
            if self.type == '4':
                a = F.upsample(fuse[0], scale_factor=8, mode='nearest')
                b = F.upsample(fuse[1], scale_factor=4, mode='nearest')
                c = F.upsample(fuse[2], scale_factor=2, mode='nearest')
                d = fuse[3]
                featureConcat = torch.cat((a, b, c, d), 1)
            else:
                a = F.upsample(fuse[0], scale_factor=4, mode='nearest')
                b = fuse[1]
                featureConcat = torch.cat((a, b), 1)
            features = self.feature_head(featureConcat)
            for f in fuse:
                cate, kernel = self.head(f)
                cates.append(cate)
                kernels.append(kernel)

        H, W, contours_count, contours_contain, seg_masks = self.head.get_HW2(features, cates, kernels, imgs_meta, self.test_cfg)
        return H, W, contours_count, contours_contain, seg_masks

    def getAllMask(self,
                   imgs,
                   gt_instance=None,
                   gt_ignore=None,
                   gt_center=None,
                   gt_centerness=None,
                   gt_dminimum=None,
                   imgs_ori=None,
                   imgs_meta=None):
        f = self.backbone(imgs)
        #### FPN
        fuse = self.neck(f[0], f[1], f[2], f[3])

        #### head
        cates = []
        kernels = []

        if self.type == '1':
            cate, kernel = self.head(fuse)
            features = self.feature_head(fuse)
            cates.append(cate)
            kernels.append(kernel)
        else:
            if self.type == '4':
                a = F.upsample(fuse[0], scale_factor=8, mode='nearest')
                b = F.upsample(fuse[1], scale_factor=4, mode='nearest')
                c = F.upsample(fuse[2], scale_factor=2, mode='nearest')
                d = fuse[3]
                featureConcat = torch.cat((a, b, c, d), 1)
            else:
                a = F.upsample(fuse[0], scale_factor=4, mode='nearest')
                b = fuse[1]
                featureConcat = torch.cat((a, b), 1)
            features = self.feature_head(featureConcat)
            for f in fuse:
                cate, kernel = self.head(f)
                cates.append(cate)
                kernels.append(kernel)

        H, W, seg_masks = self.head.get_all_mask(features, cates, kernels, imgs_meta, self.test_cfg)
        return H, W, seg_masks
