#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:50:56 2022

@author: griffo1
"""

import os


class Setup:
    def __init__(
            self,
            split=[0.6, 0.2, 0.2],
            dataset_percent=0.1,
            model_type="unet",
            depth=4,
            pool_size=2,
            concat_all=True,
            node_type=4,
            image_size=(16, 16),
            b_fil=2,
            kernel_size=3,
            dropout_amount=0.3,
            use_bn=True,
            sample_weight=None,
            batch_size=40,
            epochs=20,
            optimizer="adam",
            loss="categorical_crossentropy",
            preds_amount=50,
            bad_preds_amount=50,
            print_options=[True, True, True, True, True, True],
            name_format=["Average", "Name"],
    ):
        # dataset related:
        self.split = split
        self.dataset_percent = dataset_percent

        # Net related:
        self.model_type = model_type
        self.depth = depth
        self.pool_size = pool_size
        self.concat_all = concat_all
        self.node_type = node_type
        self.image_size = image_size
        self.input_shape = self.image_size + (1,)
        self.b_fil = b_fil
        self.kernel_size = kernel_size
        self.dropout_amount = dropout_amount
        self.use_bn = use_bn

        # fit related:
        self.sample_weight = sample_weight
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss

        # analysis related:
        self.preds_amount = preds_amount
        self.bad_preds_amount = bad_preds_amount

        # print options: [raw, output, input, input_original, gt, gt_original]
        self.print_options = print_options
        self.name_format = name_format


class NetSetup:
    def __init__(
            self,
            model_type="unet",
            depth=4,
            pool_size=2,
            concat_all=True,
            node_type=4,
            image_size=(16, 16),
            b_fil=2,
            kernel_size=3,
            dropout_amount=0.3,
            use_bn=True,
            sample_weight=None,
    ):
        # Net related:
        self.model_type = model_type
        self.depth = depth
        self.pool_size = pool_size
        self.concat_all = concat_all
        self.node_type = node_type
        self.image_size = image_size
        self.input_shape = self.image_size + (1,)
        self.b_fil = b_fil
        self.kernel_size = kernel_size
        self.dropout_amount = dropout_amount
        self.use_bn = use_bn
        self.sample_weight = sample_weight
