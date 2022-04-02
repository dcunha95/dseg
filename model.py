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
from dseg.visualize import QualityAssurance
import dseg.nets as nets
import dseg.preprocessing as prep


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

        self.trn_gen = None
        self.val_gen = None
        self.tst_gen = None
        self.stt_gen = None

        self.metrics = [tf.keras.metrics.MeanIoU(num_classes=self.setup.net_config.label_amount)]

        self.history = None
        self.callbacks = None
        self.model = None

        self.epoch_results = None

        self.data = None
        self.data_info = None
        self.dataf = None
        self.data_infof = None
        self.data_sorted = None

        self.b_ds_ready = False
        self.b_fitted = False
        self.b_analysed = False

        # create folder and update model name
        k = 0
        self.model_name = self.model_name + "_" + str(k)
        while os.path.exists(self.model_name):
            k += 1
            n = [i + "_" for i in self.model_name.split(sep="_")[:-1]]
            name = ""
            for i in n:
                name += i
            self.model_name = name + str(k)

        os.makedirs(self.model_name)

        # build model (net)
        if self.setup.model_from_file is None:

            if self.setup.net_config.model_type == "unet":
                self.model = nets.unet_14(net_config=self.setup.net_config)

            if self.setup.net_config.model_type == "unet++":
                self.model = nets.unet_pp_13(net_config=self.setup.net_config)

            if self.setup.net_config.model_type == "old_unet":
                self.model = nets.unet_4(
                    input_shape=self.setup.net_config.input_shape,
                    base_filters=self.setup.net_config.base_filters,
                    kernel_size=self.setup.net_config.kernel_size,
                    dropout_amount=self.setup.net_config.dropout_amount,
                    label_amount=3,
                    node_type=4,
                    use_bn=self.setup.net_config.use_bn,
                )

            if self.setup.net_config.model_type == "old_unet++":
                self.model = nets.unet_pp_10(
                    input_shape=self.setup.net_config.input_shape,
                    base_filters=self.setup.net_config.base_filters,
                    kernel_size=self.setup.net_config.kernel_size,
                    dropout_amount=self.setup.net_config.dropout_amount,
                    label_amount=3,
                    node_type=4,
                    use_bn=self.setup.net_config.use_bn,
                )
        # load existing model (net)
        else:
            self.model = tf.keras.models.load_model(
                self.setup.model_from_file,
                custom_objects={"get_iou_loss": QualityAssurance.get_iou_loss},
            )
            self.b_fitted = True

        # setup callbacks
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_name + "/model.h5", save_best_only=True, monitor=self.setup.fit_config.monitor, verbose=1
            )
        ]

        # setup learning rate scheduler
        if self.setup.fit_config.lr_decay_after_epoch is not None:

            def scheduler(
                epoch,
                lr,
            ):
                if epoch < self.setup.fit_config.lr_decay_after_epoch:
                    return lr
                else:
                    return lr * tf.math.exp(-self.setup.fit_config.lr_decay)

            self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1))

        # x and y preprocessing
        self.prep_x = prep.Prep.prep_x
        self.prep_y = prep.Prep.prep_y

    def get_ds(self, ds, use_tf_data: bool = True):
        """
        Builds the dataset objects.

        :param ds:
        :param use_tf_data:
        :return:
        """

        self.trn_dataset = ds[0]
        self.val_dataset = ds[1]
        self.tst_dataset = ds[2]
        self.stt_dataset = ds[3]

        min_bs = min(len(ds[0]), len(ds[1]), len(ds[2]), len(ds[3]), self.setup.fit_config.batch_size)

        if min_bs != self.setup.fit_config.batch_size:
            self.setup.fit_config.batch_size = min_bs
            print("batch_size too big, changing to ", min_bs, sep="")

        if use_tf_data:

            image_size = self.setup.net_config.image_size
            batch_size = self.setup.fit_config.batch_size

            self.trn_gen = prep.Prep.get_tf_dataset(
                ds=self.trn_dataset,
                image_size=image_size,
                batch_size=batch_size,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
            )

            self.val_gen = prep.Prep.get_tf_dataset(
                # ds=self.val_dataset,
                ds=self.stt_dataset,
                image_size=image_size,
                batch_size=batch_size,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
            )

            self.tst_gen = prep.Prep.get_tf_dataset(
                ds=self.tst_dataset,
                image_size=image_size,
                batch_size=batch_size,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
            )

            self.stt_gen = prep.Prep.get_tf_dataset(
                ds=self.stt_dataset,
                image_size=image_size,
                batch_size=batch_size,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
            )

        else:
            self.trn_gen = man.KerasManager(
                self.setup.fit_config.batch_size,
                self.setup.net_config.image_size,
                self.trn_dataset,
            )

            self.val_gen = man.KerasManager(
                self.setup.fit_config.batch_size,
                self.setup.net_config.image_size,
                self.val_dataset,
            )

            self.tst_gen = man.KerasManager(
                self.setup.fit_config.batch_size,
                self.setup.net_config.image_size,
                self.tst_dataset,
            )

            self.stt_gen = man.KerasManager(
                10,
                self.setup.net_config.image_size,
                self.stt_dataset,
            )

        self.b_ds_ready = True

    def compile(self, fit_config=None):
        """
        Compiles the model.
        
        :param fit_config: 
        """ ""

        if fit_config is None:
            fit_config = self.setup.fit_config

        if fit_config.optimizer == "adam":
            opt = tf.keras.optimizers.Adam(
                learning_rate=fit_config.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                amsgrad=False,
            )

            if tf.keras.mixed_precision.global_policy().name == "mixed_float16":
                print("Using tf.keras.mixed_precision.LossScaleOptimizer")
                opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)

        if fit_config.loss == "iou":
            loss = QualityAssurance.get_iou_loss
        else:
            loss = fit_config.loss

        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=self.metrics,
        )

    def fit(self, fit_config=None):
        """Train the model with the train dataset partition"""

        if fit_config is None:
            fit_config = self.setup.fit_config

        self.history = self.model.fit(
            self.trn_gen,
            epochs=fit_config.epochs,
            validation_data=self.val_gen,
            callbacks=self.callbacks,
        )

        self.model.save(self.model_name + "/model.h5")

        self.b_fitted = True

    def predict(
        self,
        data: pd.DataFrame = None,
        save_folder: str = "",
        verbose: int = 1,
        run_analysis: bool = True,
    ):
        """
        Retrieve predictions on data.

        :param data: DataFrame with .png image paths on column "Input Image", optionally with corresponding "Ground Truth" image paths.
        :param save_folder: Folder to save output to.
        :param verbose: Verbosity.
        :param run_analysis: Run analysis on prediction quality (only is "Ground Truth" is available).
        :return:
        """

        if data is None:
            print("No data passed")
            return

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        if not os.path.exists(save_folder + "/predictions"):
            os.makedirs(save_folder + "/predictions")

        # generate ds from data
        idx = pd.IndexSlice
        if ("Ground Truth" in data.columns) and run_analysis:
            dataset = data.loc[:, idx["Input Image", "Ground Truth"]]
            dataset.columns = pd.Index(data=["raw_path", "mask_path"])
        else:
            dataset = data.loc[:, idx["Input Image"]]
            dataset.columns = pd.Index(data=["raw_path"])

        # get tf_ds
        gen = prep.Prep.get_tf_dataset(
            ds=dataset,
            image_size=self.setup.net_config.image_size,
            batch_size=1,
            shard=False,
            prep_x=self.prep_x,
            prep_y=self.prep_y,
            return_y=run_analysis,
        )

        amount = len(gen)

        for (i, (x, y)) in enumerate(gen):
            if verbose == 1:
                print(str(i + 1) + " / " + str(amount))

            pred = self.model.predict(x)

            for (j, w) in enumerate(pred):
                if run_analysis:
                    name = vis.VisualizerAssist.pred_name(data, i, 0, name_format=self.setup.pipeline_config.name_format)
                    input_img_path = data.iloc[i + j]["Input Image"]
                    target_img_path = data.iloc[i + j]["Ground Truth"]

                    vis.VisualizerAssist.save_output(
                        name=name,
                        pred=w,
                        save_folder=save_folder,
                        input_img_path=input_img_path,
                        target_img_path=target_img_path,
                        print_options=self.setup.pipeline_config.print_options,
                    )
                else:
                    name = vis.VisualizerAssist.pred_name(data, i, 0, name_format=['Name'])
                    input_img_path = data.iloc[i + j]["Input Image"]
                    target_img_path = data.iloc[i + j]["Ground Truth"]

                    vis.VisualizerAssist.save_output(
                        name=name,
                        pred=w,
                        save_folder=save_folder,
                        input_img_path=input_img_path,
                        target_img_path=target_img_path,
                        print_options=self.setup.pipeline_config.print_options,
                    )

    def update_trainable_params(self, state):

        possible_states = {
            "unet": ["train_all", "fine_tuning"],
            "unet++": ["train_all", "fine_tuning", "hold_backbone", "train_outer_net"],
            "ivus-unet++": [],
        }

        if state not in possible_states[self.setup.net_config.model_type]:
            raise ValueError("Error updating trainable parameters: inappropriate state trying to be set.")

        for layer in self.model.layers:
            layer.trainable = False

        if state == "train_all":
            for layer in self.model.layers:
                layer.trainable = True

        if state == "fine_tuning":
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.Model):
                    for sublayer in layer.layers:
                        if isinstance(sublayer, tf.keras.layers.BatchNormalization):
                            sublayer.trainable = False
                        else:
                            sublayer.trainable = True

                elif isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False
                else:
                    layer.trainable = True

        if state == "hold_backbone":
            for layer in self.model.layers:
                layer.trainable = True

            for i in range(self.setup.net_config.depth):
                name = "bm_" + str(i) + "_0"
                self.model.get_layer(name=name).trainable = False

        if state == "train_outer_net":
            for i in range(self.setup.net_config.depth):
                name_left = "bm_" + str(i) + "_0"
                name_right = "bm_" + str(i) + "_" + str(self.setup.net_config.depth - i - 1)
                self.model.get_layer(name=name_left).trainable = True
                self.model.get_layer(name=name_right).trainable = True

    def run_analysis(self, plot_training=True):
        """Run analysis on model quality."""

        if not self.b_fitted:
            raise ValueError("Model not fitted")

        # print("\n\nEvaluating model " + self.model_name + ":\n\n")
        # results = self.model.evaluate(self.tst_gen)
        # print("\nEvaluation results:", results)

        if plot_training:
            self.epoch_results = vis.plot_training(self.history, self.model_name)

        print("\n\nRetrieving model statistics:\n\n")

        # self.data, self.data_info = QualityAssurance.retrieve_stats2(model=self.model, stat_gen=self.stt_gen, dataset=self.stt_dataset)
        self.data, self.data_info = QualityAssurance.retrieve_stats2(
            model=self.model,
            dataset=self.stt_dataset,
            image_size=self.setup.net_config.image_size,
        )

        (self.dataf, self.data_infof, self.data_sorted,) = QualityAssurance.format_table(
            data=self.data,
            data_info=self.data_info,
            dataset=self.stt_dataset,
            save_folder=self.model_name,
        )

        self.b_analysed = True

    def save_model_plot(
        self,
        show_shapes=False,
        show_dtype=False,
        show_layer_names=True,
        rankdir="TB",
    ):
        """

        :param show_shapes:
        :param show_dtype:
        :param show_layer_names:
        :param rankdir:
        :return:
        """
        tf.keras.utils.plot_model(
            model=self.model,
            to_file=self.model_name + "/model_expanded.png",
            show_shapes=show_shapes,
            rankdir=rankdir,
            show_layer_names=show_layer_names,
            show_dtype=show_dtype,
            expand_nested=True,
            dpi=96,
        )

        tf.keras.utils.plot_model(
            model=self.model,
            to_file=self.model_name + "/model.png",
            show_shapes=show_shapes,
            rankdir=rankdir,
            show_layer_names=show_layer_names,
            show_dtype=show_dtype,
            expand_nested=False,
            dpi=96,
        )

        bmodel = nets.get_base(
            input_shape=self.setup.net_config.input_shape,
            level=0,
            base_filters=self.setup.net_config.base_filters,
            kernel_size=self.setup.net_config.kernel_size,
            dropout_amount=self.setup.net_config.dropout_amount,
            node_type=self.setup.net_config.node_type,
            use_bn=self.setup.net_config.use_bn,
            name="bm",
        )

        tf.keras.utils.plot_model(
            model=bmodel,
            to_file=self.model_name + "/model_base.png",
            show_shapes=show_shapes,
            rankdir=rankdir,
            show_layer_names=show_layer_names,
            show_dtype=show_dtype,
            expand_nested=True,
            dpi=192,
        )

    def save_metrics(self):

        if not self.b_analysed:
            raise ValueError("Model not analysed.")

        self.dataf.to_csv(self.model_name + "/metrics.csv")
        self.data_infof.to_csv(self.model_name + "/metrics_summary.csv")
        self.data_sorted.to_csv(self.model_name + "/metrics_sorted.csv")

    def save_plots(self):

        if not self.b_analysed:
            raise ValueError("Model not analysed.")

        vis.VisualizerAssist.save_plots(data=self.data, data_info=self.data_info, save_folder=self.model_name, dpi=400, ci=None)

    def save_examples(
        self,
        preds_amount=20,
        bad_preds_amount=20,
    ):
        print("Saving examples")
        if not self.b_analysed:
            raise ValueError("Model not analysed")

        print("\n\nRetrieving", preds_amount, "predictions:\n\n")

        QualityAssurance.save_preds(
            model=self.model,
            data=self.dataf,
            save_folder=self.model_name,
            image_size=self.setup.net_config.image_size,
            amount=preds_amount,
            name_format=self.setup.pipeline_config.name_format,
            print_options=self.setup.pipeline_config.print_options,
            verbose=1,
        )

        print("\n\nRetrieving", bad_preds_amount, "worst predictions:\n\n")

        QualityAssurance.save_preds(
            model=self.model,
            data=self.data_sorted,
            save_folder=self.model_name,
            image_size=self.setup.net_config.image_size,
            amount=bad_preds_amount,
            name_format=self.setup.pipeline_config.name_format,
            print_options=self.setup.pipeline_config.print_options,
            verbose=1,
        )

    def save(self):
        pass
