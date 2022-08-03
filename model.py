#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Union
from itertools import product


import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

from dseg.setup import PipelineConfig, FitConfig, NetConfig, Setup
from dseg.data_manipulator import DataUtils, TrainingUtils, PlotUtils, GraphMaker
from dseg.nets import NetBuilder


class Model:
    def __init__(self, setup: Setup = Setup(), model_name: str = "model"):
        self.__model_name = model_name
        self.__setup = setup

        self.__trn_dataset = None
        self.__val_dataset = None
        self.__tst_dataset = None
        self.__stt_dataset = None

        self.__dataset_ready = False
        self.__fitted = False
        self.__analysed = False

        self.__history = None

        self.__prep_kwargs = {}

        # metrics setup
        if self.setup.net_config.multi_output:
            self.__metrics = {
                "lumen": [tf.keras.metrics.MeanIoU(num_classes=self.setup.net_config.label_amount)],
                "vessel": [tf.keras.metrics.MeanIoU(num_classes=self.setup.net_config.label_amount)],
            }
        else:
            self.__metrics = [tf.keras.metrics.MeanIoU(num_classes=self.setup.net_config.label_amount)]

        self.__analysis = None

        # update name and create folder
        self.__model_name = DataUtils.update_model_name(self.__model_name)
        
        if self.__setup.model_from_file is None:

            if self.__setup.net_config.model_type == "unet":
                self.__model = NetBuilder.unet(net_config=self.setup.net_config)

            if self.__setup.net_config.model_type == "unet++":
                self.__model = NetBuilder.unet_pp(net_config=self.setup.net_config)

        # load existing model (net)
        else:
            self.__model = tf.keras.models.load_model(
                self.setup.model_from_file,
                custom_objects={"iou_loss": TrainingUtils.iou_loss},
            )
            self.__fitted = True

        # save number of parameters
        self.setup.model_properties["parameters_number"] = self.parameters_number

        # setup callbacks
        self.__callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.__model_name + "/model.h5", save_best_only=True, monitor=self.setup.fit_config.monitor, verbose=0
            )
        ]

        # setup learning rate scheduler
        if self.setup.fit_config.lr_decay_after_epoch is not None:

            scheduler = TrainingUtils.get_scheduler_function(
                threshold=self.setup.fit_config.lr_decay_after_epoch,
                decay=self.setup.fit_config.lr_decay,
            )

            self.__callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0))

        # x and y preprocessing
        if self.setup.net_config.channels == 1:
            self.__prep_x = TrainingUtils.prep_x

        else:
            print("----> Using 2.5D strategy")
            # self.__prep_x = lambda file_path, image_size: TrainingUtils.multi_x(
            #     file_path=file_path, 
            #     channels=self.setup.net_config.channels,
            #     strides=self.setup.net_config.channel_strides,
            #     image_size=image_size,
            # )
            self.__prep_x = TrainingUtils.multi_x
            # self.__prep_kwargs["channels"] = self.setup.net_config.channels
            # self.__prep_kwargs["channel_strides"] = self.setup.net_config.channel_strides
            

        if self.setup.net_config.multi_output:
            self.__prep_y = TrainingUtils.split_y
        else:
            self.__prep_y = TrainingUtils.prep_y

        # generators (tf.data)
        self.__trn_gen = None
        self.__val_gen = None
        self.__tst_gen = None
        self.__stt_gen = None

        # save setup to json
        self.__setup.to_json("setup.json")

    @property
    def model_name(self):
        """Model name."""
        return self.__model_name

    @property
    def setup(self):
        """Setup Object."""
        return self.__setup

    @property
    def trn_dataset(self):
        """Dataset used for training."""
        return self.__trn_dataset

    @property
    def val_dataset(self):
        """Dataset used for validation."""
        return self.__val_dataset

    @property
    def tst_dataset(self):
        """Dataset used for testing."""
        return self.__tst_dataset

    @property
    def stt_dataset(self):
        """Dataset used for statistics retrieval."""
        return self.__stt_dataset

    @property
    def dataset_ready(self):
        """True if model is prepared for training, False otherwise."""
        return self.__dataset_ready

    @property
    def fitted(self):
        """True if model has been trained, False otherwise."""
        return self.__fitted

    @property
    def analysed(self):
        """True if model has had statistics calculated, False otherwise."""
        return self.__analysed

    @property
    def prep_x(self):
        """Pre-processing routine for inputs."""
        return self.__prep_x

    @property
    def prep_y(self):
        """Pre-processing routine for ground truths."""
        return self.__prep_y

    @property
    def analysis(self):
        """Dictionary containing analysis results"""
        return self.__analysis

    @property
    def parameters_number(self):
        """Return number of parameters"""
        return self.__model.count_params()

    def get_dataset(
        self,
        trn: Union[str, pd.DataFrame] = "../dataset/dataset_1MC_train.csv",
        val: Union[str, pd.DataFrame] = "../dataset/dataset_1MC_val.csv",
        tst: Union[str, pd.DataFrame] = "../dataset/dataset_1MC_test.csv",
        stt: Union[str, pd.DataFrame] = "../dataset/dataset_1MC_stat.csv",
        tf_data: bool = True,
    ):
        """
        Loads dataset partitions. Must be either the dataset reference DataFrame itself or the path to it.

        :param trn:
        :param val:
        :param tst:
        :param stt:
        :param tf_data:
        :return:
        """

        # if it is a path load the dataset, otherwise receive it
        if isinstance(trn, str):
            self.__trn_dataset = DataUtils.load_dataset_reference(trn)
        else:
            self.__trn_dataset = trn.copy()

        if isinstance(val, str):
            self.__val_dataset = DataUtils.load_dataset_reference(val)
        else:
            self.__val_dataset = val.copy()

        if isinstance(tst, str):
            self.__tst_dataset = DataUtils.load_dataset_reference(tst)
        else:
            self.__tst_dataset = tst.copy()

        if isinstance(stt, str):
            self.__stt_dataset = DataUtils.load_dataset_reference(stt)
        else:
            self.__stt_dataset = stt.copy()

        ds = self.__trn_dataset, self.__val_dataset, self.__tst_dataset, self.__stt_dataset

        min_batch_size = min(len(ds[0]), len(ds[1]), len(ds[2]), len(ds[3]), self.setup.fit_config.batch_size)

        if min_batch_size != self.setup.fit_config.batch_size:
            self.setup.fit_config.batch_size = min_batch_size

        if tf_data:
            image_size = self.setup.net_config.image_size
            batch_size = self.setup.fit_config.batch_size
            channels = self.setup.net_config.channels
            strides = self.setup.net_config.channel_strides
            

            self.__trn_gen = TrainingUtils.get_tf_dataset(
                ds=self.trn_dataset,
                image_size=image_size,
                batch_size=batch_size,
                channels=channels,
                strides=strides,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
                # **self.__prep_kwargs,
            )

            self.__val_gen = TrainingUtils.get_tf_dataset(
                ds=self.val_dataset,
                # ds=self.stt_dataset,
                image_size=image_size,
                batch_size=batch_size,
                channels=channels,
                strides=strides,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
                # **self.__prep_kwargs,
            )

            self.__tst_gen = TrainingUtils.get_tf_dataset(
                ds=self.tst_dataset,
                image_size=image_size,
                batch_size=batch_size,
                channels=channels,
                strides=strides,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
                # **self.__prep_kwargs,
            )

            self.__stt_gen = TrainingUtils.get_tf_dataset(
                ds=self.stt_dataset,
                image_size=image_size,
                batch_size=batch_size,
                channels=channels,
                strides=strides,
                prep_x=self.prep_x,
                prep_y=self.prep_y,
                return_y=True,
                # **self.__prep_kwargs,
            )

            self.__dataset_ready = True

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
            loss = TrainingUtils.iou_loss
        else:
            loss = fit_config.loss

        loss_weights = None
        if self.setup.net_config.multi_output:
            loss = {
                "lumen": loss,
                "vessel": loss,
            }
            loss_weights = {
                "lumen": 0.5,
                "vessel": 0.5,
            }

        self.__model.compile(
            optimizer=opt,
            loss=loss,
            metrics=self.__metrics,
            loss_weights=loss_weights,
        )

    def summary(self, expand_nested=False):
        """Prints neural network structure"""

        if self.__model is not None:
            self.__model.summary(expand_nested=expand_nested)


    def fit(self, fit_config=None):
        """Train the model with the train dataset partition"""

        if fit_config is None:
            fit_config = self.setup.fit_config

        self.__history = self.__model.fit(
            self.__trn_gen,
            epochs=fit_config.epochs,
            validation_data=self.__val_gen,
            callbacks=self.__callbacks,
        )

        self.__model.save(self.model_name + "/model.h5")

        self.__fitted = True

    def predict(
        self,
        data: pd.DataFrame,
        save_folder: str = "",
        verbose: int = 1,
        simple_print: bool = True,
    ):
        """
        Retrieve predictions on data.

        :param data: DataFrame with .png image paths on column "Input Image", optionally with corresponding "Ground Truth" image paths.
        :param save_folder: Folder to save output to.
        :param verbose: Verbosity.
        :param simple_print:
        :return:
        """

        if save_folder == "":
            save_folder = self.model_name

        print(save_folder, self.setup.pipeline_config.print_options)
        # make appropriate save_folder/print_options[i] paths if they don't already exist
        for option in self.setup.pipeline_config.print_options:
            path = os.path.join(save_folder, "predictions", option)
            DataUtils.make_path(path) 

        # get available columns
        dataset = data.copy()
        col_dict = {
            "raw_path": "raw_path",
            "Input Image": "raw_path",
            "mask_path": "mask_path",
            "Ground Truth": "mask_path",
            "file_name": "file_name",
            "Name": "file_name",
            "iou_avg": "iou_avg",
            "Average": "iou_avg",

            ("Path", "File Name"): "file_name",
            ("Path", "Input Image"): "raw_path",
            ("Path", "Ground Truth"): "mask_path",
        }

        cols = []
        for i in data.columns:
            if i in col_dict:
                dataset[col_dict[i]] = data.loc[:, [i]]
                cols = [col_dict[i], *cols]

        dataset = dataset[cols]

        # if not already available, get name from raw_path:
        if "file_name" not in data.columns:
            dataset["file_name"] = dataset.raw_path.str.split("\\").apply(lambda x: x[-1]).str.split(".").apply(lambda x: x[0])

        gen = TrainingUtils.get_tf_dataset(
            ds=dataset,
            image_size=self.setup.net_config.image_size,
            batch_size=1,
            channels=self.setup.net_config.channels,
            strides=self.setup.net_config.channel_strides,
            shard=True,
            prep_x=self.prep_x,
            prep_y=self.prep_y,
            return_y=False,
            # **self.__prep_kwargs,
        )

        # print options: [raw, output, input, input_original, gt, gt_original]
        
        if simple_print:
            print_options = DataUtils.print_options_to_array(["output"])
            name_format = ["Name"]
        else:
            print_options = DataUtils.print_options_to_array(self.setup.pipeline_config.print_options)
            name_format = [*self.setup.pipeline_config.name_format]

        # predict all
        amount = len(gen)
        for (i, x) in enumerate(gen):
                
            pred = self.__model.predict(x)

            # if i < 1:
            #     print("shape: ", np.array(pred).shape)

            if self.setup.net_config.multi_output:
                lumen, vessel = pred
                pred = [TrainingUtils.join_y(lumen=lumen, vessel=vessel)]
                # print(lumen.shape)
                # print(vessel.shape)
                # print(pred[0].shape)

            for (j, w) in enumerate(pred):
                name = PlotUtils.pred_name(data, i, 0, name_format=name_format)
                input_img_path = dataset.raw_path.iloc[i + j]

                # check if ground truth is available
                if "mask_path" in dataset.columns:
                    target_img_path = dataset.mask_path.iloc[i + j]
                else:
                    target_img_path = ""

                PlotUtils.save_output(
                    name=name,
                    pred=w,
                    save_folder=save_folder,
                    input_img_path=input_img_path,
                    target_img_path=target_img_path,
                    print_options=print_options,
                    image_size=self.setup.net_config.image_size,
                    x=x,
                )
            
            DataUtils.simple_counter(i, amount)


    def update_trainable_params(self, state: str = "train_all"):

        possible_states = {
            "unet": ["train_all", "fine_tuning"],
            "unet++": ["train_all", "fine_tuning", "hold_backbone", "train_outer_net"],
            "ivus-unet++": [],
        }

        if state not in possible_states[self.setup.net_config.model_type]:
            raise ValueError("Error updating trainable parameters: inappropriate state trying to be set.")

        for layer in self.__model.layers:
            layer.trainable = False

        if state == "train_all":
            for layer in self.__model.layers:
                layer.trainable = True

        if state == "fine_tuning":
            for layer in self.__model.layers:
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
            for layer in self.__model.layers:
                layer.trainable = True

            for i in range(self.setup.net_config.depth):
                name = "bm_" + str(i) + "_0"
                self.__model.get_layer(name=name).trainable = False

        if state == "train_outer_net":
            for i in range(self.setup.net_config.depth):
                name_left = "bm_" + str(i) + "_0"
                name_right = "bm_" + str(i) + "_" + str(self.setup.net_config.depth - i - 1)
                self.__model.get_layer(name=name_left).trainable = True
                self.__model.get_layer(name=name_right).trainable = True

    def plot_training(self, save_folder=None):
        """
        Saves training values and plot to .csv file, returns values in DataFrame format.

        :param save_folder:
        :return: DataFrame
        """

        if not self.__fitted or self.__history is None:
            raise ValueError("Can't plot training, not yet fitted or training info not available.")

        if save_folder is None:
            save_folder = self.model_name

        training = pd.DataFrame()

        training["Loss"] = self.__history.history["loss"]
        training["Val. Loss"] = self.__history.history["val_loss"]

        if self.setup.net_config.multi_output:

            training["Lumen Loss"] = self.__history.history["lumen_loss"]
            training["Lumen Val. Loss"] = self.__history.history["val_lumen_loss"]

            training["Vessel Loss"] = self.__history.history["vessel_loss"]
            training["Vessel Val. Loss"] = self.__history.history["val_vessel_loss"]

            training["Lumen Mean IoU"] = self.__history.history["lumen_mean_io_u"]
            training["Lumen Val. Mean IoU"] = self.__history.history["val_lumen_mean_io_u"]

            training["Vessel Mean IoU"] = self.__history.history["vessel_mean_io_u_1"]
            training["Vessel Val. Mean IoU"] = self.__history.history["val_vessel_mean_io_u_1"]

            plots = [
                ("lumen_training_results.png", ["Lumen Loss", "Lumen Val. Loss", "Lumen Mean IoU", "Lumen Val. Mean IoU"]),
                ("vessel_training_results.png", ["Vessel Loss", "Vessel Val. Loss", "Vessel Mean IoU", "Vessel Val. Mean IoU"]),
                ("training_results.png", ["Loss", "Val. Loss", "Lumen Val. Mean IoU", "Vessel Val. Mean IoU"]),
            ]

        else:

            training["Mean IoU"] = self.__history.history["mean_io_u"]
            training["Val. Mean IoU"] = self.__history.history["val_mean_io_u"]

            plots = [
                ("training_results.png", ["Loss", "Val. Loss", "Mean IoU", "Val. Mean IoU"]),
            ]

        training["Epoch"] = np.arange(1, len(training) + 1)
        training.set_index("Epoch", inplace=True)

        sns.set_theme(style="ticks")

        for plot in plots:
            graph = sns.lineplot(data=training[plot[1]], palette="bright", markers=True, dashes=False)
            graph.set(
                xlim=(1, len(training)),
                ylim=(0, 1),
                ylabel="Metric",
                xticks=np.arange(1, len(training) + 1),
            )

            # h, l = graph.get_legend_handles_labels()
            # # plt.legend(h, l, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            #
            # plt.legend(h, l, bbox_to_anchor=(1.05, 1), borderaxespad=0.0)

            graph.tick_params(axis="x", which="major", labelsize=7, rotation=45)
            graph.tick_params(axis="y", which="major", labelsize=8, rotation=0)
            graph.get_figure().savefig(save_folder + "/" + plot[0], format="png", dpi=800)
            graph.get_figure().clf()

        training.to_csv(save_folder + "/training_results.csv")

        return training

    def run_analysis(self, ref_data: pd.DataFrame = None, save_folder: str = "", verbose: bool = True):
        """
        Run prediction quality analysis routine on reference data. If none passed, will run on stats dataset.

        :param ref_data:
        :param save_folder:
        :param verbose:
        :return:
        """

        if not self.__fitted:
            raise ValueError("Model not fitted")

        if save_folder == "":
            save_folder = self.model_name

        if ref_data is None:
            ref_data = self.__stt_dataset

        DataUtils.make_path(save_folder)

        # if not os.path.exists(save_folder + "/predictions"):
        #     os.makedirs(save_folder + "/predictions")

        # get available columns
        dataset = ref_data.copy()
        col_dict = {
            "raw_path": "raw_path",
            "Input Image": "raw_path",
            "mask_path": "mask_path",
            "Ground Truth": "mask_path",
            "file_name": "file_name",
            "Name": "file_name",
            "iou_avg": "iou_avg",
            "Average": "iou_avg",
            
            ("Path", "File Name"): "file_name",
            ("Path", "Input Image"): "raw_path",
            ("Path", "Ground Truth"): "mask_path",
        }

        cols = []
        for i in ref_data.columns:
            if i in col_dict:
                dataset[col_dict[i]] = ref_data.loc[:, [i]]
                cols = [col_dict[i], *cols]

        dataset = dataset[cols]

        # if not already available, get name from raw_path:
        if "file_name" not in ref_data.columns:
            dataset["file_name"] = dataset.raw_path.str.split("\\").apply(lambda x: x[-1]).str.split(".").apply(lambda x: x[0])

        gen = TrainingUtils.get_tf_dataset(
            ds=dataset,
            image_size=self.setup.net_config.image_size,
            batch_size=1,
            channels=self.setup.net_config.channels,
            strides=self.setup.net_config.channel_strides,
            shard=True,
            prep_x=self.prep_x,
            prep_y=self.prep_y,
            return_y=True,
            # **self.__prep_kwargs,
        )

        # print_options = self.setup.pipeline_config.print_options

        # predict all
        amount = len(gen)
        stats_list = [None for k in range(amount)]
        for (i, (x, y)) in enumerate(gen):

            
            pred = self.__model.predict(x)

            if self.setup.net_config.multi_output:
                lumen, vessel = pred
                pred = [TrainingUtils.join_y(lumen=lumen, vessel=vessel)]
                # print(lumen.shape)
                # print(vessel.shape)
                # print(pred[0].shape)

            for (j, w) in enumerate(pred):
                # name = PlotUtils.pred_name(ref_data, i, 0, name_format=["iou_avg", "file_name"])
                # input_img_path = dataset.iloc[i + j]["raw_path"]
                target_img_path = dataset.iloc[i + j]["mask_path"]

                stats_list[i + j] = TrainingUtils.prediction_metrics(prediction=w, target_img_path=target_img_path)

                # PlotUtils.save_output(
                #     name=name,
                #     pred=w,
                #     save_folder=save_folder,
                #     input_img_path=input_img_path,
                #     target_img_path=target_img_path,
                #     print_options=print_options,
                # )

            DataUtils.simple_counter(i, amount)
            

        data = pd.DataFrame(
            np.array(stats_list, dtype="float32"),
        )

        data.columns = pd.MultiIndex.from_tuples([
                *product(["IoU", "DICE"], ["Average", "Outer", "Lumen", "Plaque", "Vessel",]),
                *product(["Hausdorf Distance [mm]"], ["Lumen", "Plaque", "Vessel",]),
                *product(["Area [mm²]"], ["Lumen", "Lumen GT", "Plaque", "Plaque GT", "Vessel", "Vessel GT",]),
                *product(["Area Ratio"], ["Lumen", "Plaque", "Vessel",]),
                *product(["Plaque Burden"], ["Prediction", "Ground Truth", "Ratio",]),
                ],
            )

        data_info = data.describe()
        data_info.loc["IQR"] = data_info.loc["75%"] - data_info.loc["25%"]
        data_info = data_info.loc[["count", "mean", "std", "IQR", "min", "25%", "50%", "75%", "max"]]

        data_formatted, data_info_formatted, data_sorted = DataUtils.format_table(
            data=data,
            data_info=data_info,
            dataset=self.__stt_dataset,
        )

        out = {
            "data_formatted": data_formatted,
            "data_info_formatted": data_info_formatted,
            "data_sorted": data_sorted,
            "data": data,
            "data_info": data_info,
        }

        self.__analysis = out

        # Graph maker
        dpi = 250
        figure_args = {
            "figsize": (4*19.20/5, 4*10.80/5),
            "dpi": dpi,
        }

        metric_translation = {
            "DICE" : "Sørensen-Dice Index", 
            "IoU": "Jaccard Index", 
        }
        
        plotter = GraphMaker(figure_args=figure_args, metric_translation=metric_translation)
        PlotUtils.standard_plot_routine(data, save_folder, plotter)

        plotter.figure_args['figsize'] =  (4*19.20/5, 2*10.80/5)
        plotter.figure_args['dpi'] = 500
        PlotUtils.standard_plot_routine(data, save_folder, plotter, kname='thin')


        self.__analysed = True


