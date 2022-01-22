#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:39:08 2021

@author: griffo1
"""

import numpy as np
import tensorflow as tf
# import processing as pr

class KerasManager(tf.keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        image_size,
        dataset,
        label_amount=3,
        sample_weight=None,
        normalize=True,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.dataset = dataset
        self.label_amount = label_amount
        self.sample_weight = sample_weight
        self.normalize = normalize

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size

        batch = self.dataset.iloc[i : i + self.batch_size]

        x = np.zeros((self.batch_size,) + self.image_size + (1,), dtype="uint8")

        j = 0
        for number, item in batch.iterrows():
            img = tf.keras.preprocessing.image.load_img(
                item.raw_path,
                color_mode="grayscale",
                target_size=self.image_size,
                interpolation="nearest",
            )
            x[j] = np.expand_dims(img, 2)
            j += 1

        x = x.astype("float64")
        if self.normalize == True:
            x = x / 255.0

        y = np.zeros(
            (self.batch_size,) + self.image_size + (self.label_amount,), dtype="uint8"
        )

        if self.sample_weight != None:
            # pesos das classes
            w = np.zeros((self.batch_size,) + self.image_size + (1,))

        j = 0
        for number, item in batch.iterrows():
            img = tf.keras.preprocessing.image.load_img(
                item.mask_path,
                color_mode="grayscale",
                target_size=self.image_size,
                interpolation="nearest",
            )
            img = np.array(img, dtype="uint16")

            # if self.sample_weight != None:
            #     img2 = np.array(img, dtype="float32")
            #     w[j, :, :, 0] = pr.mark_edges(7, img2)

            # quantas labels serao utilizadas?
            if self.label_amount == 2:
                y[j, :, :, 0] = (img == 0).astype("uint8")
                y[j, :, :, 1] = (img == 255).astype("uint8")

            else:
                y[j, :, :, 0] = (img == 0).astype("uint8")
                y[j, :, :, 1] = (img == 100).astype("uint8")
                y[j, :, :, 2] = (img == 255).astype("uint8")

            j += 1

        y = y.astype("float64")

        if self.sample_weight != None:

            w = w * (self.sample_weight - 1) + 1
            return x, y, w

        else:
            return x, y
