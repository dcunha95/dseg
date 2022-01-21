#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 15:33:32 2022

@author: griffo1
"""

import tensorflow as tf
import pandas as pd
import seaborn as sns
import os

import dseg.manager as man
import dseg.visualize as vis
import dseg.nets as nets

# from IPython.display import Image


class Segmenter:
    def __init__(
        self,
        setup,
        model_name="ivus_seg",
    ):
        self.model_name = model_name
        self.setup = setup
        self.trn_dataset = None
        self.val_dataset = None
        self.tst_dataset = None
        self.stt_dataset = None

        if self.setup.model_type == "unet":
            self.model = nets.unet_12(
                input_shape=self.setup.input_shape,
                b_fil=self.setup.b_fil,
                kernel_size=self.setup.kernel_size,
                dropout_amount=self.setup.dropout_amount,
                label_amount=3,
                node_type=self.setup.node_type,
                use_bn=self.setup.use_bn,
                depth=self.setup.depth,
                pool_size=self.setup.pool_size,
            )

        if self.setup.model_type == "unet++":
            self.model = nets.unet_pp_11(
                input_shape=self.setup.input_shape,
                b_fil=self.setup.b_fil,
                kernel_size=self.setup.kernel_size,
                dropout_amount=self.setup.dropout_amount,
                label_amount=3,
                node_type=self.setup.node_type,
                use_bn=self.setup.use_bn,
                pool_size=self.setup.pool_size,
                concat_all=self.setup.concat_all,
            )

        self.model.compile(
            optimizer=self.setup.optimizer,
            loss=self.setup.loss,
            metrics=[tf.keras.metrics.MeanIoU(num_classes=3)],
        )

        #### create folder and update model name
        k = 0
        self.model_name = self.model_name + "_" + str(k)
        while os.path.exists(self.model_name) == True:
            k += 1
            n = [i + "_" for i in self.model_name.split(sep="_")[:-1]]
            name = ""
            for i in n:
                name += i
            self.model_name = name + str(k)

        os.makedirs(self.model_name)
        ####

        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_name + "/model.h5",
                save_best_only=True,
                monitor="val_mean_io_u",
            ),
        ]

        self.iou_list = None

    def get_ds(self, ds):
        self.trn_dataset = ds[0]
        self.val_dataset = ds[1]
        self.tst_dataset = ds[2]
        self.stt_dataset = ds[3]

        self.trn_gen = man.KerasManager(
            self.setup.batch_size,
            self.setup.image_size,
            self.trn_dataset,
        )

        self.val_gen = man.KerasManager(
            self.setup.batch_size,
            self.setup.image_size,
            self.val_dataset,
        )

        self.tst_gen = man.KerasManager(
            self.setup.batch_size,
            self.setup.image_size,
            self.val_dataset,
        )

        self.stt_gen = man.KerasManager(
            10,
            self.setup.image_size,
            self.stt_dataset,
        )

    def fit(self):

        self.history = self.model.fit(
            self.trn_gen,
            epochs=self.setup.epochs,
            validation_data=self.val_gen,
            callbacks=self.callbacks,
        )

        self.model.save(self.model_name + "/model.h5")

    def plot_model(
        self,
        save=True,
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
        expand_nested=False,
        subgraph=False,
    ):

        # plot = Image(
        #     tf.keras.utils.model_to_dot(
        #         self.model,
        #         show_shapes=show_shapes,
        #         show_dtype=show_dtype,
        #         show_layer_names=show_layer_names,
        #         rankdir=rankdir,
        #         expand_nested=expand_nested,
        #         dpi=96,
        #         subgraph=False,
        #     ).create_png()
        # )

        if save:
            tf.keras.utils.plot_model(
                model=self.model,
                to_file=self.model_name + "/model_expanded.png",
                show_shapes=show_shapes,
                rankdir=rankdir,
                show_layer_names=show_layer_names,
                show_dtype=show_dtype,
                expand_nested=True,
                dpi=192,
            )

            tf.keras.utils.plot_model(
                model=self.model,
                to_file=self.model_name + "/model.png",
                show_shapes=show_shapes,
                rankdir=rankdir,
                show_layer_names=show_layer_names,
                show_dtype=show_dtype,
                expand_nested=False,
                dpi=192,
            )

        # return plot

    def analisys(self, preds_amount=None, bad_preds_amount=None):

        if preds_amount == None:
            preds_amount = self.setup.preds_amount

        if bad_preds_amount == None:
            bad_preds_amount = self.setup.bad_preds_amount

        print("\n\nEvaluating model " + self.model_name + ":\n\n")
        results = self.model.evaluate(self.tst_gen)
        print("\nEvaluation results:", results)

        self.epochs = vis.plot_training(self.history, self.model_name)

        print("\n\nRetrieving model statistics:\n\n")

        self.iou_list = vis.retrieve_stats(
            model=self.model,
            stat_gen=self.stt_gen,
            dataset=self.stt_dataset,
        )

        self.data, self.data_info = vis.get_statistics(self.iou_list, self.model_name)

        self.data_pretty, self.data_info_pretty, data_sorted = vis.format_table(
            self.data,
            self.data_info,
            self.stt_dataset,
            self.model_name,
            return_formatted=True,
        )

        print("\n\nRetrieving", preds_amount, "predictions:\n\n")

        vis.save_preds(
            model=self.model,
            stat_gen=self.stt_gen,
            data=self.data_pretty,
            save_folder=self.model_name,
            amount=preds_amount,
            name_format=self.setup.name_format,
            print_options=self.setup.print_options,
            verbose=1,
        )

        print("\n\nRetrieving", bad_preds_amount, "worst predictions:\n\n")

        vis.get_bad_preds(
            model=self.model,
            data_sorted=data_sorted,
            save_folder=self.model_name,
            image_size=self.setup.image_size,
            amount=bad_preds_amount,
            name_format=self.setup.name_format,
            verbose=1,
        )

    def save(self):

        pass
