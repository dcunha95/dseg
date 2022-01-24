#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 16:34:46 2021

@author: griffo1
"""

import tensorflow as tf
import numpy as np
import os
import random
import PIL
import tensorflow.keras.layers as ly


#%% UNET


def unet_4(
    input_shape,
    b_fil,
    kernel_size,
    dropout_amount=0.2,
    label_amount=3,
    node_type=1,
    use_bn=False,
):

    inputs = tf.keras.Input(input_shape)

    x = inputs

    def base_node(
        x,
        level,
        b_fil=b_fil,
        kernel_size=kernel_size,
        dropout_amount=dropout_amount,
        node_type=node_type,
        use_bn=use_bn,
    ):

        # micro-estrutura base
        if node_type == 1:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura dupla
        if node_type == 2:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura paralela
        if node_type == 3:

            # 1
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node1)
            if use_bn == True:
                node1 = ly.BatchNormalization()(node1)
            else:
                node1 = ly.Dropout(dropout_amount)(node1)

            # 2
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node2)
            if use_bn == True:
                node2 = ly.BatchNormalization()(node2)
            else:
                node2 = ly.Dropout(dropout_amount)(node2)

            node = ly.concatenate([node1, node2])

        # conv, bn, relu
        if node_type == 4:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        return node

    ## Downsampling

    x00 = base_node(x, 0)

    x10 = ly.MaxPool2D(pool_size=2, padding="same")(x00)
    x10 = base_node(x10, 1)

    x20 = ly.MaxPool2D(pool_size=2, padding="same")(x10)
    x20 = base_node(x20, 2)

    x30 = ly.MaxPool2D(pool_size=2, padding="same")(x20)
    x30 = base_node(x30, 3)

    ## Upsampling

    x21 = ly.UpSampling2D(2)(x30)
    x21 = ly.concatenate([x21, x20])
    x21 = base_node(x21, 2)

    x11 = ly.UpSampling2D(2)(x21)
    x11 = ly.concatenate([x11, x10])
    x11 = base_node(x11, 1)

    x01 = ly.UpSampling2D(2)(x11)
    x01 = ly.concatenate([x01, x00])

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(x01)

    model = tf.keras.Model(inputs, outputs, name="unet_4")
    return model


def unet_5(
    input_shape,
    b_fil,
    kernel_size,
    dropout_amount=0.2,
    label_amount=3,
    node_type=1,
    use_bn=False,
):

    inputs = tf.keras.Input(input_shape)

    x = inputs

    def base_node(
        x,
        level,
        b_fil=b_fil,
        kernel_size=kernel_size,
        dropout_amount=dropout_amount,
        node_type=node_type,
        use_bn=use_bn,
    ):

        # micro-estrutura base
        if node_type == 1:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura dupla
        if node_type == 2:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura paralela
        if node_type == 3:

            # 1
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node1)
            if use_bn == True:
                node1 = ly.BatchNormalization()(node1)
            else:
                node1 = ly.Dropout(dropout_amount)(node1)

            # 2
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node2)
            if use_bn == True:
                node2 = ly.BatchNormalization()(node2)
            else:
                node2 = ly.Dropout(dropout_amount)(node2)

            node = ly.concatenate([node1, node2])

        # conv, bn, relu
        if node_type == 4:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        return node

    ## Downsampling

    x00 = base_node(x, 0)

    x10 = ly.MaxPool2D(pool_size=2, padding="same")(x00)
    x10 = base_node(x10, 1)

    x20 = ly.MaxPool2D(pool_size=2, padding="same")(x10)
    x20 = base_node(x20, 2)

    x30 = ly.MaxPool2D(pool_size=2, padding="same")(x20)
    x30 = base_node(x30, 3)

    x40 = ly.MaxPool2D(pool_size=2, padding="same")(x30)
    x40 = base_node(x40, 4)

    ## Upsampling

    x31 = ly.UpSampling2D(2)(x40)
    x31 = ly.concatenate([x31, x30])
    x31 = base_node(x31, 3)

    x21 = ly.UpSampling2D(2)(x31)
    x21 = ly.concatenate([x21, x20])
    x21 = base_node(x21, 2)

    x11 = ly.UpSampling2D(2)(x21)
    x11 = ly.concatenate([x11, x10])
    x11 = base_node(x11, 1)

    x01 = ly.UpSampling2D(2)(x11)
    x01 = ly.concatenate([x01, x00])

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(x01)

    model = tf.keras.Model(inputs, outputs, name="unet_5")
    return model


def unet_6(
    input_shape,
    b_fil,
    kernel_size,
    dropout_amount=0.2,
    label_amount=3,
    node_type=1,
    use_bn=False,
):

    inputs = tf.keras.Input(input_shape)

    x = inputs

    def base_node(
        x,
        level,
        b_fil=b_fil,
        kernel_size=kernel_size,
        dropout_amount=dropout_amount,
        node_type=node_type,
        use_bn=use_bn,
    ):

        # micro-estrutura base
        if node_type == 1:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura dupla
        if node_type == 2:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura paralela
        if node_type == 3:

            # 1
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node1)
            if use_bn == True:
                node1 = ly.BatchNormalization()(node1)
            else:
                node1 = ly.Dropout(dropout_amount)(node1)

            # 2
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node2)
            if use_bn == True:
                node2 = ly.BatchNormalization()(node2)
            else:
                node2 = ly.Dropout(dropout_amount)(node2)

            node = ly.concatenate([node1, node2])

        # conv, bn, relu
        if node_type == 4:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        return node

    ## Downsampling

    x00 = base_node(x, 0)

    x10 = ly.MaxPool2D(pool_size=2, padding="same")(x00)
    x10 = base_node(x10, 1)

    x20 = ly.MaxPool2D(pool_size=2, padding="same")(x10)
    x20 = base_node(x20, 2)

    x30 = ly.MaxPool2D(pool_size=2, padding="same")(x20)
    x30 = base_node(x30, 3)

    x40 = ly.MaxPool2D(pool_size=2, padding="same")(x30)
    x40 = base_node(x40, 4)

    x50 = ly.MaxPool2D(pool_size=2, padding="same")(x40)
    x50 = base_node(x50, 5)

    ## Upsampling

    x41 = ly.UpSampling2D(2)(x50)
    x41 = ly.concatenate([x41, x40])
    x41 = base_node(x41, 4)

    x31 = ly.UpSampling2D(2)(x41)
    x31 = ly.concatenate([x31, x30])
    x31 = base_node(x31, 3)

    x21 = ly.UpSampling2D(2)(x31)
    x21 = ly.concatenate([x21, x20])
    x21 = base_node(x21, 2)

    x11 = ly.UpSampling2D(2)(x21)
    x11 = ly.concatenate([x11, x10])
    x11 = base_node(x11, 1)

    x01 = ly.UpSampling2D(2)(x11)
    x01 = ly.concatenate([x01, x00])

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(x01)

    model = tf.keras.Model(inputs, outputs, name="unet_6")
    return model


#%% UNET++


def unet_pp_10(
    input_shape,
    b_fil,
    kernel_size,
    dropout_amount=0.2,
    label_amount=3,
    node_type=0,
    use_bn=False,
):

    inputs = tf.keras.Input(input_shape)
    x = inputs

    def base_node(
        x,
        level,
        b_fil=b_fil,
        kernel_size=kernel_size,
        dropout_amount=dropout_amount,
        node_type=node_type,
        use_bn=use_bn,
    ):

        # micro-estrutura padrao
        if node_type == 0:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)

        # micro-estrutura base
        if node_type == 1:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura dupla
        if node_type == 2:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura paralela
        if node_type == 3:

            # 1
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node1)
            if use_bn == True:
                node1 = ly.BatchNormalization()(node1)
            else:
                node1 = ly.Dropout(dropout_amount)(node1)

            # 2
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node2)
            if use_bn == True:
                node2 = ly.BatchNormalization()(node2)
            else:
                node2 = ly.Dropout(dropout_amount)(node2)

            node = ly.concatenate([node1, node2])

        # conv, bn, relu
        if node_type == 4:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        # (conv, bn, relu)x2
        if node_type == 5:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(node)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        return node

    # Backbone

    # x00
    x00 = base_node(x, 0)

    # x10
    x10 = ly.MaxPool2D(pool_size=2, padding="same")(x00)
    x10 = base_node(x10, 1)
    # x20
    x20 = ly.MaxPool2D(pool_size=2, padding="same")(x10)
    x20 = base_node(x20, 2)
    # x30
    x30 = ly.MaxPool2D(pool_size=2, padding="same")(x20)
    x30 = base_node(x30, 3)
    # x40
    x40 = ly.MaxPool2D(pool_size=2, padding="same")(x30)
    x40 = base_node(x40, 4)
    # x50
    x50 = ly.MaxPool2D(pool_size=2, padding="same")(x40)
    x50 = base_node(x50, 5)

    # In-betweens

    # x01
    x01 = ly.concatenate([x00, ly.UpSampling2D(2)(x10)])
    x01 = base_node(x01, 0)
    # x11
    x11 = ly.concatenate([x10, ly.UpSampling2D(2)(x20)])
    x11 = base_node(x11, 1)
    # x21
    x21 = ly.concatenate([x20, ly.UpSampling2D(2)(x30)])
    x21 = base_node(x21, 2)
    # x31
    x31 = ly.concatenate([x30, ly.UpSampling2D(2)(x40)])
    x31 = base_node(x31, 3)
    # x41
    x41 = ly.concatenate([x40, ly.UpSampling2D(2)(x50)])
    x41 = base_node(x41, 4)
    # x02
    x02 = ly.concatenate([x00, x01, ly.UpSampling2D(2)(x11)])
    x02 = base_node(x02, 0)
    # x12
    x12 = ly.concatenate([x10, x11, ly.UpSampling2D(2)(x21)])
    x12 = base_node(x12, 1)
    # x22
    x22 = ly.concatenate([x20, x21, ly.UpSampling2D(2)(x31)])
    x22 = base_node(x22, 2)
    # x32
    x32 = ly.concatenate([x30, x31, ly.UpSampling2D(2)(x41)])
    x32 = base_node(x32, 3)
    # x03
    x03 = ly.concatenate([x00, x01, x02, ly.UpSampling2D(2)(x12)])
    x03 = base_node(x03, 0)
    # x13
    x13 = ly.concatenate([x10, x11, x12, ly.UpSampling2D(2)(x22)])
    x13 = base_node(x13, 1)
    # x23
    x23 = ly.concatenate([x20, x21, x22, ly.UpSampling2D(2)(x32)])
    x23 = base_node(x23, 2)
    # x04
    x04 = ly.concatenate([x00, x01, x02, x03, ly.UpSampling2D(2)(x13)])
    x04 = base_node(x04, 0)
    # x14
    x14 = ly.concatenate([x10, x11, x12, x13, ly.UpSampling2D(2)(x23)])
    x14 = base_node(x14, 1)
    # x05
    x05 = ly.concatenate([x00, x01, x02, x03, x04, ly.UpSampling2D(2)(x14)])
    x05 = base_node(x05, 0)
    # x06
    x06 = ly.concatenate(
        [x00, x01, x02, x03, x04, x05, ly.UpSampling2D(2)(x14)]
    )
    x06 = base_node(x06, 0)

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(x06)

    model = tf.keras.Model(inputs, outputs, name="unet++_10")
    return model


#%% generalized unet


def unet_11(
    input_shape,
    b_fil,
    kernel_size,
    depth=5,
    dropout_amount=0.2,
    label_amount=3,
    node_type=0,
    use_bn=False,
):

    inputs = tf.keras.Input(input_shape)
    x = inputs

    def base_node(
        x,
        level,
        b_fil=b_fil,
        kernel_size=kernel_size,
        dropout_amount=dropout_amount,
        node_type=node_type,
        use_bn=use_bn,
    ):

        # micro-estrutura padrao
        if node_type == 0:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)

        # micro-estrutura base
        if node_type == 1:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura dupla
        if node_type == 2:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura paralela
        if node_type == 3:

            # 1
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node1 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node1)
            if use_bn == True:
                node1 = ly.BatchNormalization()(node1)
            else:
                node1 = ly.Dropout(dropout_amount)(node1)

            # 2
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(x)
            node2 = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
            )(node2)
            if use_bn == True:
                node2 = ly.BatchNormalization()(node2)
            else:
                node2 = ly.Dropout(dropout_amount)(node2)

            node = ly.concatenate([node1, node2])

        # conv, bn, relu
        if node_type == 4:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        # (conv, bn, relu)x2
        if node_type == 5:
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(x)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)
            node = ly.Conv2D(
                filters=b_fil * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
            )(node)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        return node

    nodes = [[] for i in range(depth)]

    # descend
    for k in range(depth):
        nodes[k].append(base_node(x, k))
        x = ly.MaxPool2D(pool_size=2, padding="same")(nodes[k][-1])

    for k in range(depth - 2, -1, -1):
        x = ly.concatenate(
            [nodes[k][-1], ly.UpSampling2D(2)(nodes[k + 1][-1])]
        )
        nodes[k].append(base_node(x, k))

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(nodes[0][-1])

    model = tf.keras.Model(inputs, outputs, name="unet_11")
    return model


#%% new


def base_node(
    x,
    level,
    b_fil,
    kernel_size,
    dropout_amount,
    node_type,
    use_bn,
):

    # micro-estrutura padrao
    if node_type == 0:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(x)

    # micro-estrutura base
    if node_type == 1:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(x)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        if use_bn == True:
            node = ly.BatchNormalization()(node)
        else:
            node = ly.Dropout(dropout_amount)(node)

    # micro-estrutura dupla
    if node_type == 2:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(x)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        if use_bn == True:
            node = ly.BatchNormalization()(node)
        else:
            node = ly.Dropout(dropout_amount)(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        if use_bn == True:
            node = ly.BatchNormalization()(node)
        else:
            node = ly.Dropout(dropout_amount)(node)

    # micro-estrutura paralela
    if node_type == 3:

        # 1
        node1 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(x)
        node1 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node1)
        if use_bn == True:
            node1 = ly.BatchNormalization()(node1)
        else:
            node1 = ly.Dropout(dropout_amount)(node1)

        # 2
        node2 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(x)
        node2 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node2)
        if use_bn == True:
            node2 = ly.BatchNormalization()(node2)
        else:
            node2 = ly.Dropout(dropout_amount)(node2)

        node = ly.concatenate([node1, node2])

    # conv, bn, relu
    if node_type == 4:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(x)
        node = ly.BatchNormalization()(node)
        node = ly.Activation("relu")(node)

    # (conv, bn, relu)x2
    if node_type == 5:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(x)
        node = ly.BatchNormalization()(node)
        node = ly.Activation("relu")(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(node)
        node = ly.BatchNormalization()(node)
        node = ly.Activation("relu")(node)

    return node


def get_base(
    input_shape,
    level,
    b_fil,
    kernel_size,
    dropout_amount,
    node_type,
    use_bn,
    name,
):

    inputs = ly.Input(input_shape)

    # micro-estrutura padrao
    if node_type == 0:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(inputs)

    # micro-estrutura base
    if node_type == 1:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(inputs)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        if use_bn == True:
            node = ly.BatchNormalization()(node)
        else:
            node = ly.Dropout(dropout_amount)(node)

    # micro-estrutura dupla
    if node_type == 2:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(inputs)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        if use_bn == True:
            node = ly.BatchNormalization()(node)
        else:
            node = ly.Dropout(dropout_amount)(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node)
        if use_bn == True:
            node = ly.BatchNormalization()(node)
        else:
            node = ly.Dropout(dropout_amount)(node)

    # micro-estrutura paralela
    if node_type == 3:

        # 1
        node1 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(inputs)
        node1 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node1)
        if use_bn == True:
            node1 = ly.BatchNormalization()(node1)
        else:
            node1 = ly.Dropout(dropout_amount)(node1)

        # 2
        node2 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(inputs)
        node2 = ly.Conv2D(
            filters=b_fil * 2 ** level,
            kernel_size=kernel_size,
            activation="relu",
            padding="same",
        )(node2)
        if use_bn == True:
            node2 = ly.BatchNormalization()(node2)
        else:
            node2 = ly.Dropout(dropout_amount)(node2)

        node = ly.concatenate([node1, node2])

    # conv, bn, relu
    if node_type == 4:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(inputs)
        node = ly.BatchNormalization()(node)
        node = ly.Activation("relu")(node)

    # (conv, bn, relu)x2
    if node_type == 5:
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(inputs)
        node = ly.BatchNormalization()(node)
        node = ly.Activation("relu")(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(node)
        node = ly.BatchNormalization()(node)
        node = ly.Activation("relu")(node)

    if node_type == 6:
        x = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(inputs)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(x)
        node = ly.Activation("relu")(node)
        node = ly.Conv2D(
            filters=b_fil * 2 ** level, kernel_size=kernel_size, padding="same"
        )(node)
        node = ly.Add()([node, x])

    outputs = node

    model = tf.keras.Model(inputs, outputs, name=name)

    return model


def unet_12(
    input_shape,
    b_fil,
    kernel_size,
    depth=5,
    dropout_amount=0.2,
    label_amount=3,
    node_type=0,
    use_bn=False,
    pool_size=2,
):

    # print("input_shape:", input_shape)
    inputs = tf.keras.Input(input_shape)
    x = inputs

    nodes = [[] for i in range(depth)]
    # descend
    for k in range(depth):
        name = "bm_" + str(k) + "_0"
        # print(name)
        # shape = (
        #     int(input_shape[0] / (pool_size ** k)),
        #     int(input_shape[1] / (pool_size ** k)),
        #     int(input_shape[2] * max(1, (b_fil * 2 ** k))),
        # )
        # print("node_entrance:", shape)
        # print("x.shape:", x.shape)

        node = get_base(
            input_shape=x.shape[1:],
            level=k,
            b_fil=b_fil,
            kernel_size=kernel_size,
            dropout_amount=dropout_amount,
            node_type=node_type,
            use_bn=use_bn,
            name=name,
        )(x)
        nodes[k].append(node)
        # print("node_exit:", node.shape)
        x = ly.MaxPool2D(pool_size=pool_size, padding="same")(nodes[k][-1])
        # print("maxpool:", x.shape)

    for k in range(depth - 2, -1, -1):
        # print(k)
        name = "bm_" + str(k) + "_1"
        x = ly.concatenate(
            [nodes[k][-1], ly.UpSampling2D(2)(nodes[k + 1][-1])]
        )

        node = get_base(
            input_shape=x.shape[1:],
            level=k,
            b_fil=b_fil,
            kernel_size=kernel_size,
            dropout_amount=dropout_amount,
            node_type=node_type,
            use_bn=use_bn,
            name=name,
        )(x)
        nodes[k].append(node)

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(nodes[0][-1])

    model = tf.keras.Model(inputs, outputs, name="unet_12")
    return model


def unet_pp_11(
    input_shape,
    b_fil,
    kernel_size,
    depth=5,
    dropout_amount=0.2,
    label_amount=3,
    node_type=0,
    use_bn=False,
    pool_size=2,
    concat_all=True,
):
    # print("input_shape:", input_shape)

    inputs = tf.keras.Input(input_shape)
    x = inputs

    nodes = [[] for i in range(depth)]

    # descend (backbone)
    for i in range(depth):
        name = "bm_" + str(i) + "_0"
        # print(name)
        # print("node_entrance:", shape)
        # print("x.shape:", x.shape)
        # shape = (
        #     int(input_shape[0] / (pool_size ** i)),
        #     int(input_shape[1] / (pool_size ** i)),
        #     b_fil * 2 ** i if i == 0 else 1,
        # )
        node = get_base(
            input_shape=x.shape[1:],
            level=i,
            b_fil=b_fil,
            kernel_size=kernel_size,
            dropout_amount=dropout_amount,
            node_type=node_type,
            use_bn=use_bn,
            name=name,
        )(x)
        # print("node_exit:", node.shape)
        nodes[i].append(node)
        x = ly.MaxPool2D(pool_size=pool_size, padding="same")(nodes[i][-1])
        # print("maxpool:", x.shape)

    for j in range(1, depth):
        for i in range(depth - j):
            # print(i, j, sep="\t")
            name = "bm_" + str(i) + "_" + str(j)
            if concat_all:
                layers = [*nodes[i]]
            else:
                layers = [nodes[i][-1]]
            layers.append(ly.UpSampling2D(pool_size)(nodes[i + 1][-1]))
            x = ly.concatenate(layers)

            node = get_base(
                input_shape=x.shape[1:],
                level=i,
                b_fil=b_fil,
                kernel_size=kernel_size,
                dropout_amount=dropout_amount,
                node_type=node_type,
                use_bn=use_bn,
                name=name,
            )(x)
            nodes[i].append(node)

    outputs = ly.Conv2D(
        filters=label_amount,
        kernel_size=kernel_size,
        activation="softmax",
        padding="same",
    )(nodes[0][-1])

    model = tf.keras.Model(inputs, outputs, name="unet_12")
    return model
