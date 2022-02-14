# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:51:39 2021

@author: danie
"""

import tensorflow as tf
import numpy as np


class GIoU:
    @staticmethod
    def get_bbox(
            img
            ):
        # rows = tf.math.reduce_any(img, axis=1)
        # cols = tf.math.reduce_any(img, axis=0)
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return rmin, rmax, cmin, cmax

    @staticmethod
    def _calculate_giou(b1, b2):
        zero = tf.convert_to_tensor(0.0, b1.dtype)
        b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)
        b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
        b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
        b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
        b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
        b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
        b1_area = b1_width * b1_height
        b2_area = b2_width * b2_height

        intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
        intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
        intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
        intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
        intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
        intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
        intersect_area = intersect_width * intersect_height

        union_area = b1_area + b2_area - intersect_area
        iou = tf.math.divide_no_nan(intersect_area, union_area)

        enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
        enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
        enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
        enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
        enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
        enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
        enclose_area = enclose_width * enclose_height
        giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area)
        return giou

    @staticmethod
    def giou_loss(y_true, y_pred):

        y_pred = tf.convert_to_tensor(y_pred)
        if not y_pred.dtype.is_floating:
            y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        giou = tf.squeeze(GIoU._calculate_giou(y_pred, y_true))

        return 1-giou
