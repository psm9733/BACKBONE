import cv2
import numpy as np
import torch
import torch.nn as nn
import math


class YoloLoss(nn.Module):
    def __init__(self, input_shape, classes, branch_num, anchor, batch_size, loss_type='diou'):
        super().__init__()
        self.input_shape = input_shape
        self.classes = classes
        self.branch_num = branch_num
        self.anchor = anchor
        self.batch_size = batch_size
        self.anchor_num = int(len(anchor) / self.branch_num)
        self.loss_type = loss_type
        self.lamda_coord = 5
        self.lamda_noobj = 0.5
        self.epsilon = 1e-7

    def euclidean_distance(self, pred, gt):
        '''
            Example:
                batch = 4, anchor_num = 3
                ->  pred =  [batch, anchor_num, 2, h, w]
                ->  gt = [batch, anchor_num, 2, h, w]
        '''
        distance = torch.sqrt(torch.pow(pred[:, :, 0:1, :, :] - gt[:, :, 0:1, :, :], 2) +
                              torch.pow(pred[:, :, 1:2, :, :] - gt[:, :, 1:2, :, :], 2) +
                              self.epsilon)
        return distance

    def get_iou(self, pred, gt):
        '''
            Example:
                batch = 4, anchor_num = 3, class = 80
                ->  pred =  [batch, anchor_num, 5 + class, h, w]
                ->  gt = [batch, anchor_num, 5 + class, h, w]
        '''
        pred_xy = pred[:, :, 1:3, :, :]
        pred_wh = pred[:, :, 3:5, :, :]
        pred_mins = pred_xy - pred_wh / 2
        pred_maxs = pred_xy + pred_wh / 2
        gt_xy = gt[:, :, 1:3, :, :]
        gt_wh = gt[:, :, 3:5, :, :]
        gt_mins = gt_xy - gt_wh / 2
        gt_maxs = gt_xy + gt_wh / 2
        intersect_min = torch.maximum(pred_mins, gt_mins)
        intersect_max = torch.minimum(pred_maxs, gt_maxs)
        intersect_wh = intersect_max - intersect_min
        intersect_wh = torch.maximum(intersect_wh, torch.zeros_like(intersect_wh))

        pred_area = pred_wh[:, :, 0:1, :, :] * pred_wh[:, :, 1:2, :, :]
        gt_area = gt_wh[:, :, 0:1, :, :] * gt_wh[:, :, 1:2, :, :]
        intersect_area = intersect_wh[:, :, 0:1, :, :] * intersect_wh[:, :, 1:2, :, :]
        union_area = pred_area + gt_area - intersect_area
        iou = intersect_area / (union_area + self.epsilon)
        return iou

    def iou_loss(self, pred, gt):
        '''
            Example:
                batch = 4, anchor_num = 3, class = 80
                ->  pred =  [batch, anchor_num, 5 + class, h, w]
                ->  gt = [batch, anchor_num, 5 + class, h, w]
        '''
        iou = self.get_iou(pred, gt)
        iou_loss = torch.ones_like(iou) - iou
        return iou_loss

    def get_iou_panalty(self, pred, gt):
        '''
            Example:
                batch = 4, anchor_num = 3, class = 80
                ->  pred =  [batch, anchor_num, 5 + class, h, w]
                ->  gt = [batch, anchor_num, 5 + class, h, w]
        '''
        pred_xy = pred[:, :, 1:3, :, :]
        pred_wh = pred[:, :, 3:5, :, :]
        pred_center = pred_wh / 2
        pred_mins = pred_xy - pred_center
        pred_maxs = pred_xy + pred_center

        gt_xy = gt[:, :, 1:3, :, :]
        gt_wh = gt[:, :, 3:5, :, :]
        gt_center = gt_wh / 2
        gt_mins = gt_xy - gt_center
        gt_maxs = gt_xy + gt_center

        big_bbox_mins = torch.minimum(pred_mins, gt_mins)
        big_bbox_maxs = torch.minimum(pred_maxs, gt_maxs)

        p = self.euclidean_distance(pred_center, gt_center)
        c = self.euclidean_distance(big_bbox_maxs, big_bbox_mins)

        panalty = torch.pow(p, 2) / (torch.pow(c, 2) + self.epsilon)
        return panalty

    def DIoULoss(self, pred, gt):
        '''
            Example:
                batch = 4, anchor_num = 3, class = 80
                ->  pred =  [batch, anchor_num, 5 + class, h, w]
                ->  gt = [batch, anchor_num, 5 + class, h, w]
        '''
        diou_loss = self.iou_loss(pred, gt) + self.get_iou_panalty(pred, gt)
        return diou_loss

    def forward(self, pred, gt):
        '''
            Example:
                batch = 4, anchor_num = 3, class = 80, branch_num = 3,  output_shape = [[76, 76], [38, 38], [19, 19]]
                ->  pred =  [[batch, anchor_num, 5 + class, 76, 76], [batch, anchor_num, 5 + class, 38, 38], [batch, anchor_num, 5 + class, 19, 19]]
                ->  gt = [[batch, anchor_num, 5 + class, 76, 76], [batch, anchor_num, 5 + class, 38, 38], [batch, anchor_num, 5 + class, 19, 19]]
        '''
        confidence_loss = 0
        location_loss = 0
        class_loss = 0
        for branch_index in range(self.branch_num):
            gt_branch = gt[branch_index]
            pred_branch = pred[branch_index]
            obj_mask = gt_branch[:, :, 0:1, :, :]
            no_obj_mask = torch.ones_like(obj_mask) - obj_mask
            pred_branch[:, :, 0:1, :, :] = torch.sigmoid(pred_branch[:, :, 0:1, :, :])
            pred_branch[:, :, 5:, :, :] = torch.sigmoid(pred_branch[:, :, 5:, :, :])
            gt_branch[:, :, 1:3, :, :] = torch.logit(gt_branch[:, :, 1:3, :, :], eps=self.epsilon)
            for anchor_index in range(self.anchor_num):
                anchor = self.anchor[self.anchor_num * (self.branch_num - branch_index - 1) + anchor_index]
                gt_branch[:, anchor_index:anchor_index + 1, 3:4, :, :] = (
                        gt_branch[:, anchor_index:anchor_index + 1, 3:4, :, :] / anchor[0] + self.epsilon)
                gt_branch[:, anchor_index:anchor_index + 1, 4:5, :, :] = (
                        gt_branch[:, anchor_index:anchor_index + 1, 4:5, :, :] / anchor[1] + self.epsilon)

            obj_loss = torch.sum(obj_mask * nn.BCELoss(reduction="none")(pred_branch[:, :, 0:1, :, :],
                                                                         gt_branch[:, :, 0:1, :, :])) / self.batch_size
            no_obj_loss = torch.sum(no_obj_mask * nn.BCELoss(reduction="none")(pred_branch[:, :, 0:1, :, :],
                                                                               gt_branch[:, :, 0:1, :,
                                                                               :])) / self.batch_size
            confidence_loss += self.lamda_coord * obj_loss + self.lamda_noobj * no_obj_loss

            xy_loss = torch.sum(obj_mask * (nn.MSELoss(reduction="none")(pred_branch[:, :, 1:3, :, :],
                                                                         gt_branch[:, :, 1:3, :, :]))) / self.batch_size
            wh_loss = torch.sum(obj_mask * (nn.MSELoss(reduction="none")(pred_branch[:, :, 3:5, :, :],
                                                                         gt_branch[:, :, 3:5, :, :]))) / self.batch_size
            location_loss += (xy_loss + wh_loss)

            class_loss += torch.sum(obj_mask * (
                nn.BCELoss(reduction="none")(pred_branch[:, :, 5:, :, :], gt_branch[:, :, 5:, :, :]))) / self.batch_size
        total_loss = confidence_loss + location_loss + class_loss
        # print("=================")
        # print("confidence_loss = ", confidence_loss.item())
        # print("location_loss = ", location_loss.item())
        # print("class_loss = ", class_loss.item())
        # print("total_loss = ", total_loss.item())
        return total_loss, confidence_loss, location_loss, class_loss
