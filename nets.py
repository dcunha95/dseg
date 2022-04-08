import tensorflow as tf
import numpy as np
import os
import random
import PIL
import tensorflow.keras.layers as ly


class NetBuilder:
    """Class responsible for building nets (the Keras model)."""

    @staticmethod
    def get_base(
        input_shape,
        level,
        base_filters,
        kernel_size,
        dropout_amount,
        node_type,
        use_bn,
        name,
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        inputs = ly.Input(input_shape)

        # micro-estrutura padrao
        if node_type == 0:
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(inputs)

        # micro-estrutura base
        if node_type == 1:
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(inputs)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura dupla
        if node_type == 2:
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(inputs)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node)
            if use_bn == True:
                node = ly.BatchNormalization()(node)
            else:
                node = ly.Dropout(dropout_amount)(node)

        # micro-estrutura paralela
        if node_type == 3:

            # 1
            node1 = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(inputs)
            node1 = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node1)
            if use_bn == True:
                node1 = ly.BatchNormalization()(node1)
            else:
                node1 = ly.Dropout(dropout_amount)(node1)

            # 2
            node2 = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(inputs)
            node2 = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                activation="relu",
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node2)
            if use_bn == True:
                node2 = ly.BatchNormalization()(node2)
            else:
                node2 = ly.Dropout(dropout_amount)(node2)

            node = ly.concatenate([node1, node2])

        # conv, bn, relu
        if node_type == 4:
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                use_bias=False,
            )(inputs)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        # (conv, bn, relu)x2
        if node_type == 5:
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                use_bias=False,
            )(inputs)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                use_bias=False,
            )(node)
            node = ly.BatchNormalization()(node)
            node = ly.Activation("relu")(node)

        if node_type == 6:
            x = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(inputs)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(x)
            node = ly.Activation("relu")(node)
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(node)
            node = ly.Add()([node, x])

        # conv, bn, prelu
        if node_type == 7:
            node = ly.Conv2D(
                filters=base_filters * 2 ** level,
                kernel_size=kernel_size,
                padding="same",
                dilation_rate=dilation_rate,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                use_bias=False,
            )(inputs)
            node = ly.BatchNormalization()(node)
            node = ly.PReLU()(node)

        outputs = node

        model = tf.keras.Model(inputs, outputs, name=name)

        return model

    @staticmethod
    def unet(net_config):
        input_shape = net_config.input_shape
        base_filters = net_config.base_filters
        kernel_size = net_config.kernel_size
        depth = net_config.depth
        dropout_amount = net_config.dropout_amount
        label_amount = net_config.label_amount
        node_type = net_config.node_type
        use_bn = net_config.use_bn
        pool_size = net_config.pool_size
        down_size = net_config.down_size
        kernel_initializer = net_config.kernel_initializer
        bias_initializer = net_config.bias_initializer

        # print("input_shape:", input_shape)
        inputs = tf.keras.Input(input_shape)
        x = inputs

        # downsize
        if down_size is not None:
            x = ly.Conv2D(filters=1, kernel_size=(down_size, down_size), strides=down_size)(x)
            # x = ly.MaxPool2D(down_size)(x)

        nodes = [[] for i in range(depth)]
        # descend
        for k in range(depth):
            name = "bm_" + str(k) + "_0"
            # print(name)
            # shape = (
            #     int(input_shape[0] / (pool_size ** k)),
            #     int(input_shape[1] / (pool_size ** k)),
            #     int(input_shape[2] * max(1, (base_filters * 2 ** k))),
            # )
            # print("node_entrance:", shape)
            # print("x.shape:", x.shape)

            node = NetBuilder.get_base(
                input_shape=x.shape[1:],
                level=k,
                base_filters=base_filters,
                kernel_size=kernel_size,
                dropout_amount=dropout_amount,
                node_type=node_type,
                use_bn=use_bn,
                name=name,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(x)
            nodes[k].append(node)
            # print("node_exit:", node.shape)
            x = ly.MaxPool2D(pool_size=pool_size, padding="same")(nodes[k][-1])
            # print("maxpool:", x.shape)

        for k in range(depth - 2, -1, -1):
            # print(k)
            name = "bm_" + str(k) + "_1"
            x = ly.concatenate([nodes[k][-1], ly.UpSampling2D(2)(nodes[k + 1][-1])])

            node = NetBuilder.get_base(
                input_shape=x.shape[1:],
                level=k,
                base_filters=base_filters,
                kernel_size=kernel_size,
                dropout_amount=dropout_amount,
                node_type=node_type,
                use_bn=use_bn,
                name=name,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
            )(x)
            nodes[k].append(node)

        outputs = ly.Conv2D(filters=label_amount, kernel_size=kernel_size, padding="same")(nodes[0][-1])

        # upsize
        if down_size is not None:
            # x = ly.Conv2D(filters=label_amount, kernel_size=(down_size, down_size), strides=down_size)(x)
            outputs = ly.UpSampling2D(down_size, interpolation="bilinear")(outputs)
            # outputs = ly.Conv2DTranspose(filters=label_amount, kernel_size=(down_size, down_size), strides=down_size)(outputs)

        outputs = ly.Activation("softmax", dtype="float32")(outputs)

        model = tf.keras.Model(inputs, outputs, name="unet_13")
        return model

    @staticmethod
    def unet_pp(net_config):
        input_shape = net_config.input_shape
        base_filters = net_config.base_filters
        kernel_size = net_config.kernel_size
        depth = net_config.depth
        dropout_amount = net_config.dropout_amount
        label_amount = net_config.label_amount
        node_type = net_config.node_type
        use_bn = net_config.use_bn
        pool_size = net_config.pool_size
        down_size = net_config.down_size
        concat_all = net_config.concat_all
        kernel_initializer = net_config.kernel_initializer
        bias_initializer = net_config.bias_initializer

        # print("input_shape:", input_shape)

        inputs = tf.keras.Input(input_shape)
        x = inputs

        # downsize
        if down_size is not None:
            # x = ly.Conv2D(filters=1, kernel_size=(down_size, down_size), strides=down_size)(x)
            x = ly.MaxPool2D(down_size)(x)

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
            #     base_filters * 2 ** i if i == 0 else 1,
            # )
            node = NetBuilder.get_base(
                input_shape=x.shape[1:],
                level=i,
                base_filters=base_filters,
                kernel_size=kernel_size,
                dropout_amount=dropout_amount,
                node_type=node_type,
                use_bn=use_bn,
                name=name,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
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
                layers.append(ly.UpSampling2D(pool_size, interpolation="bilinear")(nodes[i + 1][-1]))
                x = ly.concatenate(layers)

                node = NetBuilder.get_base(
                    input_shape=x.shape[1:],
                    level=i,
                    base_filters=base_filters,
                    kernel_size=kernel_size,
                    dropout_amount=dropout_amount,
                    node_type=node_type,
                    use_bn=use_bn,
                    name=name,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                )(x)
                nodes[i].append(node)

        outputs = ly.Conv2D(filters=label_amount, kernel_size=kernel_size, padding="same")(nodes[0][-1])

        # upsize
        if down_size is not None:
            # x = ly.Conv2D(filters=label_amount, kernel_size=(down_size, down_size), strides=down_size)(x)
            outputs = ly.UpSampling2D(down_size, interpolation="bilinear")(outputs)
            # outputs = ly.Conv2DTranspose(filters=label_amount, kernel_size=(down_size, down_size), strides=down_size)(outputs)

        outputs = ly.Activation("softmax", dtype="float32")(outputs)

        model = tf.keras.Model(inputs, outputs, name="unet_pp_12")
        return model



