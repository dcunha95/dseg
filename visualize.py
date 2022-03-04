# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:51:39 2021

@author: danie
"""

import numpy as np
import tensorflow as tf
import PIL
import os
from PIL import ImageOps
import pandas as pd
import seaborn as sns
import matplotlib
import dseg.manager as man
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")


class QualityAssurance:
    @staticmethod
    def get_iou(y_pred, y_true):
        """Returns IoU (Jaccard Index)"""
        inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
        union = np.count_nonzero(np.logical_or(y_pred, y_true).astype("uint8"))
        return inter / union

    @staticmethod
    def get_dice(y_pred, y_true):
        """Returns DICE (Sørensen–Dice Index)"""
        inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
        x = np.count_nonzero(y_pred.astype("uint8"))
        y = np.count_nonzero(y_true.astype("uint8"))
        return inter / (x + y)

    @staticmethod
    def get_iou_loss(y_true, y_pred):
        inter = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[1, 2])
        union = tf.math.reduce_sum(y_true + y_pred, axis=[1, 2]) - inter
        iou_loss = 1 - tf.math.reduce_mean(tf.math.divide_no_nan(inter, union), axis=1)

        return iou_loss

    @staticmethod
    def get_iou_squared_loss(y_true, y_pred):
        inter = tf.math.reduce_sum(tf.math.pow(tf.math.multiply(y_pred, y_true), 2), axis=[1, 2])
        union = tf.math.reduce_sum(y_true + y_pred, axis=[1, 2]) - inter
        iou_loss = 1 - tf.math.reduce_mean(tf.math.divide_no_nan(inter, union), axis=1)

        return iou_loss

    @staticmethod
    def prediction_metrics(prediction, target_img_path, use_original=False):
        """Take quality metrics from a prediction"""

        pred = np.argmax(prediction, axis=-1)

        if use_original:
            gt = tf.keras.preprocessing.image.load_img(
                target_img_path,
                color_mode="grayscale",
            )
            pred = tf.keras.preprocessing.image.array_to_img(np.expand_dims(pred, axis=-1))
            pred = pred.resize((500, 500))
            pred = tf.keras.preprocessing.image.img_to_array(pred)
            pred = np.squeeze(pred)
        else:
            gt = tf.keras.preprocessing.image.load_img(
                target_img_path,
                color_mode="grayscale",
                target_size=prediction.shape[:-1],
                interpolation="nearest",
            )

        gt = tf.keras.preprocessing.image.img_to_array(gt, dtype="uint8")
        gt = np.squeeze(gt)

        # splitting each class
        pred_e = (pred == 0).astype("uint8")
        pred_l = (pred == 1).astype("uint8")
        pred_p = (pred == 2).astype("uint8")
        pred_v = pred_l + pred_p

        gt_e = (gt == 0).astype("uint8")
        gt_l = (gt == 100).astype("uint8")
        gt_p = (gt == 255).astype("uint8")
        gt_v = gt_l + gt_p

        # area
        tot_area = prediction.shape[0] ** 2

        area_pred_l = np.count_nonzero(pred_l) / tot_area
        area_pred_p = np.count_nonzero(pred_p) / tot_area
        area_pred_v = np.count_nonzero(pred_v) / tot_area

        area_gt_l = np.count_nonzero(gt_l) / tot_area
        area_gt_p = np.count_nonzero(gt_p) / tot_area
        area_gt_v = np.count_nonzero(gt_v) / tot_area

        # error_area_l = area_pred_l - area_gt_l
        # error_area_p = area_pred_p - area_gt_p
        # error_area_v = area_pred_v - area_gt_v

        # Calculating IoUs

        # external, lumen, plaque and vessel
        # (vessel = lumen + placa)
        iou_e = QualityAssurance.get_iou(pred_e, gt_e)
        iou_l = QualityAssurance.get_iou(pred_l, gt_l)
        iou_p = QualityAssurance.get_iou(pred_p, gt_p)
        iou_v = QualityAssurance.get_iou(pred_v, gt_v)

        # average iou between plaque and lumen
        iou_avg = (iou_l + iou_p) / 2

        # # Calculating DICE
        # dice_e = QualityAssurance.get_dice(pred_e, gt_e)
        # dice_l = QualityAssurance.get_dice(pred_l, gt_l)
        # dice_p = QualityAssurance.get_dice(pred_p, gt_p)
        # dice_v = QualityAssurance.get_dice(pred_v, gt_v)
        #
        # # average DICE between plaque and lumen
        # dice_avg = (dice_l + dice_p) / 2

        metrics = [
            iou_avg,
            iou_e,
            iou_l,
            iou_p,
            iou_v,
            area_pred_l,
            area_gt_l,
            area_pred_p,
            area_gt_p,
            area_pred_v,
            area_gt_v,
        ]

        return metrics

    @staticmethod
    def retrieve_stats(
        model,
        stat_gen,
        dataset,
        verbose=1,
    ):
        """Retrieve quality statistics based on stat dataset"""
        loops = stat_gen.__len__() + 1

        # initialize the array that will hold metrics information
        stats_list = [None for k in range(len(dataset))]

        for i in range(loops):

            if verbose == 1:
                print(str(i + 1) + " / " + str(loops))
            batch = stat_gen.__getitem__(i)[0]

            # if on last loop, not all batchs might be full
            if i == loops - 1:
                batch = batch[: (len(dataset) - stat_gen.batch_size * stat_gen.__len__())]
            preds = model.predict_on_batch(batch)

            # calculate base position on arrays
            base_j = i * stat_gen.batch_size

            for j in range(len(preds)):

                stats_list[j + base_j] = QualityAssurance.prediction_metrics(preds[j], dataset.mask_path.iloc[j + base_j])

        data = pd.DataFrame(
            np.array(stats_list, dtype="float32"),
            columns=[
                "Average",
                "Outer",
                "Lumen",
                "Plaque",
                "Vessel",
                "Lumen Area [mm²]",
                "Lumen Area GT [mm²]",
                "Plaque Area [mm²]",
                "Plaque Area GT [mm²]",
                "Vessel Area [mm²]",
                "Vessel Area GT [mm²]",
            ],
        )

        # More metrics
        data["Lumen Area Ratio"] = data["Lumen Area [mm²]"] / data["Lumen Area GT [mm²]"]
        data["Plaque Area Ratio"] = data["Plaque Area [mm²]"] / data["Plaque Area GT [mm²]"]
        data["Vessel Area Ratio"] = data["Vessel Area [mm²]"] / data["Vessel Area GT [mm²]"]

        # A single image has 100mm2
        data.loc[:, "Lumen Area [mm²]":"Vessel Area GT [mm²]"] = 100 * data.loc[:, "Lumen Area [mm²]":"Vessel Area GT [mm²]"]

        # Even more
        data["Plaque Burden"] = data["Plaque Area [mm²]"] / data["Vessel Area [mm²]"]
        data["Plaque Burden GT"] = data["Plaque Area GT [mm²]"] / data["Vessel Area GT [mm²]"]

        # MOOOAAAAAR
        data["Plaque Burden Model/GT Ratio"] = data["Plaque Burden"] / data["Plaque Burden GT"]

        data_info = data.describe()

        # data.to_csv(save_folder + "/metrics.csv")
        # data_info.to_csv(save_folder + "/metrics_summary.csv")

        return data, data_info

    @staticmethod
    def format_table(
        data,
        data_info,
        dataset,
        save_folder,
        sorting_metric="Average",
    ):
        """
        Formats data and data_info.

        :param data:
        :param data_info:
        :param dataset:
        :param save_folder:
        :param sorting_metric:
        :return:
        """
        dataf = data.loc[:, :].astype("object")
        data_infof = data_info.loc[:, :].astype("object")

        data_infof.loc["count"] = data_infof.loc["count"].map("{:.0f}".format)

        for i in dataf.iteritems():
            dataf.loc[:, i[0]] = dataf[i[0]].map("{:.4f}".format)
            data_infof.loc["mean":"max", i[0]] = data_infof.loc["mean":"max", i[0]].map("{:.4f}".format)

        dataf = dataf.astype("str")
        data_infof = data_infof.astype("str")

        df = dataset.set_index(np.arange(len(dataset)))

        dataf.insert(0, "Name", df.file_name)

        dataf.Name = dataf.Name.apply(lambda x: x[:-4])

        dataf["Input Image"] = dataset.set_index(np.arange(len(dataset))).raw_path
        dataf["Ground Truth"] = dataset.set_index(np.arange(len(dataset))).mask_path

        dataf.to_csv(save_folder + "/metrics_pretty.csv")
        data_infof.to_csv(save_folder + "/metrics_summary_pretty.csv")

        data_sorted = dataf.sort_values(by=sorting_metric)

        # data_sorted.to_csv(save_folder + "/metrics_sorted.csv")

        return dataf, data_infof, data_sorted


class VisualizerAssist:
    @staticmethod
    def pred_name(data, j, base_j, name_format=["Average", "Name"]):
        """Returns a name prediction."""
        name = ""

        for i in name_format:
            name += data.iloc[j + base_j][i] + "_"

        name += str(data.index[base_j + j])
        name = name.replace(".", "")

        return name

    @staticmethod
    def save_output(
        name,
        pred,
        input_img_path,
        target_img_path,
        save_folder,
        print_options=[True, True, True, True, True, True],
    ):

        if print_options[0]:
            img = PIL.ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(pred))
            img.save(save_folder + "/predictions" + "/" + name + "_raw.png", format="png")
        if print_options[1]:
            img = np.array(np.argmax(pred, axis=-1), dtype="uint8")
            img = np.expand_dims((img == 1).astype("uint8") * 100 + (img == 2).astype("uint8") * 255, axis=-1)
            img = tf.keras.preprocessing.image.array_to_img(img, scale=False)
            img.save(save_folder + "/predictions" + "/" + name + "_output.png", format="png")
        if print_options[2]:
            img = tf.keras.preprocessing.image.load_img(
                input_img_path, color_mode="grayscale", target_size=pred.shape, interpolation="nearest"
            )
            img.save(save_folder + "/predictions" + "/" + name + "_input.png", format="png")
        if print_options[3]:
            img = tf.keras.preprocessing.image.load_img(input_img_path)
            img.save(save_folder + "/predictions" + "/" + name + "_input_original.png", format="png")
        if print_options[4]:
            img = tf.keras.preprocessing.image.load_img(
                target_img_path, color_mode="grayscale", target_size=pred.shape, interpolation="nearest"
            )
            img.save(save_folder + "/predictions" + "/" + name + "_gt.png", format="png")
        if print_options[5]:
            img = tf.keras.preprocessing.image.load_img(target_img_path)
            img.save(save_folder + "/predictions" + "/" + name + "_gt_original.png", format="png")
        return

    @staticmethod
    def ratio_plots(
        name,
        gt_name,
        ratio_name,
        file_name,
        data,
        data_info,
        plots_folder,
        dpi=96,
        ci=None,
    ):
        """

        :param name:
        :param gt_name:
        :param ratio_name:
        :param file_name:
        :param data:
        :param data_info:
        :param plots_folder:
        :param dpi:
        :param ci:
        :return:
        """
        sns.set_theme(style="whitegrid")

        # [min x, max x, min y, max y]
        data_max = max(data_info.loc["max", [name, gt_name]])

        graph = sns.scatterplot(x=data[gt_name], y=data[name], marker="o", color="red", s=3)
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph.get_figure().savefig(plots_folder + "/" + file_name + ".png", format="png", dpi=dpi, bbox_inches="tight")
        graph.get_figure().clf()

        graph = sns.scatterplot(x=data[gt_name], y=data[name], marker="o", color="red", s=3)
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.axis("scaled")
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph.get_figure().savefig(plots_folder + "/" + "scaled_" + file_name + ".png", format="png", dpi=dpi, bbox_inches="tight")
        graph.get_figure().clf()

        graph = matplotlib.pyplot.axes()
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph = sns.regplot(data=data, x=gt_name, y=name, color="red", truncate=False, ci=ci, scatter_kws={"s": 3})
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.get_figure().savefig(plots_folder + "/" + "regression_" + file_name + ".png", format="png", dpi=dpi, bbox_inches="tight")
        graph.get_figure().clf()

        graph = matplotlib.pyplot.axes()
        graph.axis("scaled")
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph = sns.regplot(data=data, x=gt_name, y=name, color="red", truncate=False, ci=ci, scatter_kws={"s": 3})
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.get_figure().savefig(
            plots_folder + "/" + "regression_scaled_" + file_name + ".png", format="png", dpi=dpi, bbox_inches="tight"
        )
        graph.get_figure().clf()

        graph = sns.scatterplot(data=data[ratio_name], marker="o", color="red", s=3)
        graph = sns.lineplot(x=[i for i in range(len(data))], y=[1 for i in range(len(data))])
        graph.set(xlim=(0, len(data)), ylim=(0, max(2, 1.1 * data_info.loc["max", ratio_name])))
        graph.get_figure().savefig(plots_folder + "/" + "ratio_" + file_name + ".png", format="png", dpi=dpi, bbox_inches="tight")
        graph.get_figure().clf()

        graph = matplotlib.pyplot.axes()
        graph.set(xlim=(0, len(data)), ylim=(0, max(2, 1.1 * data_info.loc["max", ratio_name])))
        graph = sns.regplot(
            data=data, x=[i for i in range(len(data))], y=data[ratio_name], color="red", truncate=False, ci=ci, scatter_kws={"s": 3}
        )
        graph = sns.lineplot(x=[i for i in range(len(data))], y=[1 for i in range(len(data))])
        graph.get_figure().savefig(
            plots_folder + "/" + "ratio_regression_" + file_name + ".png", format="png", dpi=dpi, bbox_inches="tight"
        )
        graph.get_figure().clf()

        return

    @staticmethod
    def save_plots(
        data,
        data_info,
        save_folder,
        dpi=400,
        ci=None,
    ):
        """
        Make plots.

        :param data:
        :param data_info:
        :param save_folder:
        :param dpi:
        :param ci:
        :return:
        """

        plots_folder = save_folder + "/plots"
        if not os.path.exists(plots_folder):
            os.makedirs(plots_folder)

        # no mean
        for bw in ["scott", 0.01, 0.1, 0.2]:
            graph = sns.violinplot(data=data.loc[:, "Lumen":"Vessel"], saturation=0.9, bw=bw, gridsize=400, cut=0)
            graph.set(ylim=(0, 1.03))
            graph.get_figure().savefig(plots_folder + "/iou_violin_bw_" + str(bw) + ".png", format="png", dpi=dpi)
            graph.get_figure().clf()

        graph = sns.boxplot(data=data.loc[:, "Lumen":"Vessel"], saturation=0.9)
        graph.set(ylim=(0, 1.03))
        graph.get_figure().savefig(plots_folder + "/iou_box.png", format="png", dpi=dpi)
        graph.get_figure().clf()

        graph = sns.scatterplot(data=data.loc[:, "Lumen":"Vessel"], markers=["o", "o", "o"], alpha=0.85, edgecolor=None)
        graph.set(xlim=(0, len(data)), ylim=(0, 1.03))
        graph.get_figure().savefig(save_folder + "/plots" + "/iou_scatter.png", format="png", dpi=dpi)
        graph.get_figure().clf()

        # with mean
        idx = pd.IndexSlice
        iou_cols = idx["Lumen", "Plaque", "Vessel", "Average"]

        for bw in ["scott", 0.01, 0.1, 0.2]:
            graph = sns.violinplot(data=data.loc[:, iou_cols], saturation=0.9, bw=bw, gridsize=400, cut=0)
            graph.set(ylim=(0, 1.03))
            graph.get_figure().savefig(plots_folder + "/iou_violin_bw_" + str(bw) + "_avg.png", format="png", dpi=dpi)
            graph.get_figure().clf()

        graph = sns.boxplot(data=data.loc[:, iou_cols], saturation=0.9)
        graph.set(ylim=(0, 1.03))
        graph.get_figure().savefig(plots_folder + "/iou_box_avg.png", format="png", dpi=dpi)
        graph.get_figure().clf()

        graph = sns.scatterplot(data=data.loc[:, "Average"], markers=["o"], alpha=0.85, edgecolor=None)
        graph.set(ylim=(0, 1.03))
        graph.set(xlim=(0, len(data)), ylim=(0, 1.03))
        graph.get_figure().savefig(save_folder + "/plots" + "/iou_scatter_avg.png", format="png", dpi=dpi)
        graph.get_figure().clf()

        VisualizerAssist.ratio_plots(
            name="Lumen Area [mm²]",
            gt_name="Lumen Area GT [mm²]",
            ratio_name="Lumen Area Ratio",
            file_name="area_lumen",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        VisualizerAssist.ratio_plots(
            name="Plaque Area [mm²]",
            gt_name="Plaque Area GT [mm²]",
            ratio_name="Plaque Area Ratio",
            file_name="area_plaque",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        VisualizerAssist.ratio_plots(
            name="Vessel Area [mm²]",
            gt_name="Vessel Area GT [mm²]",
            ratio_name="Vessel Area Ratio",
            file_name="area_vessel",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )

        VisualizerAssist.ratio_plots(
            name="Plaque Area [mm²]",
            gt_name="Vessel Area [mm²]",
            ratio_name="Plaque Burden",
            file_name="plaque_burden",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        VisualizerAssist.ratio_plots(
            name="Plaque Area GT [mm²]",
            gt_name="Vessel Area GT [mm²]",
            ratio_name="Plaque Burden GT",
            file_name="plaque_burden_gt",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        VisualizerAssist.ratio_plots(
            name="Plaque Burden",
            gt_name="Plaque Burden GT",
            ratio_name="Plaque Burden Model/GT Ratio",
            file_name="plaque_burden_model_gt_comparison",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )

        return


#%%

###################################################################

# define prediction file name
def pred_name(data, j, base_j, name_format=["Average", "Name"]):
    name = ""

    for i in name_format:
        name += data.iloc[j + base_j][i] + "_"
    name += str(data.index[base_j + j])
    name = name.replace(".", "")

    return name


# salva uma unica predicao
def save_output(
    name,
    pred,
    input_img_path,
    target_img_path,
    save_folder,
    print_options=[True, True, True, True, True, True],
):
    if print_options[0]:
        img = PIL.ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(pred))
        img.save(save_folder + "/predictions" + "/" + name + "_raw.png", format="png")

    if print_options[1]:
        img = np.array(np.argmax(pred, axis=-1), dtype="uint8")
        img = np.expand_dims((img == 1).astype("uint8") * 100 + (img == 2).astype("uint8") * 255, axis=-1)
        img = tf.keras.preprocessing.image.array_to_img(img, scale=False)
        img.save(save_folder + "/predictions" + "/" + name + "_output.png", format="png")

    if print_options[2]:
        img = tf.keras.preprocessing.image.load_img(input_img_path, color_mode="grayscale", target_size=pred.shape, interpolation="nearest")
        img.save(save_folder + "/predictions" + "/" + name + "_input.png", format="png")

    if print_options[3]:
        img = tf.keras.preprocessing.image.load_img(input_img_path)
        img.save(save_folder + "/predictions" + "/" + name + "_input_original.png", format="png")

    if print_options[4]:
        img = tf.keras.preprocessing.image.load_img(
            target_img_path, color_mode="grayscale", target_size=pred.shape, interpolation="nearest"
        )
        img.save(save_folder + "/predictions" + "/" + name + "_gt.png", format="png")

    if print_options[5]:
        img = tf.keras.preprocessing.image.load_img(target_img_path)
        img.save(save_folder + "/predictions" + "/" + name + "_gt_original.png", format="png")

    return


def get_iou(y_pred, y_true):
    """
    :param y_pred:
    :param y_true:
    :return:
    """
    inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
    union = np.count_nonzero(np.logical_or(y_pred, y_true).astype("uint8"))
    return inter / union


# take statistics from prediction
def save_stats(prediction, target_img_path, use_original=False):
    # tomando a predicao e a ground truth
    pred = np.argmax(prediction, axis=-1)

    if use_original:
        gt = tf.keras.preprocessing.image.load_img(
            target_img_path,
            color_mode="grayscale",
        )
        pred = tf.keras.preprocessing.image.array_to_img(np.expand_dims(pred, axis=-1))
        pred = pred.resize((500, 500))
        pred = tf.keras.preprocessing.image.img_to_array(pred)
        pred = np.squeeze(pred)
    else:
        gt = tf.keras.preprocessing.image.load_img(
            target_img_path,
            color_mode="grayscale",
            target_size=prediction.shape[:-1],
            interpolation="nearest",
        )
    gt = tf.keras.preprocessing.image.img_to_array(gt, dtype="uint8")
    gt = np.squeeze(gt)

    # separando cada classe
    pred_e = (pred == 0).astype("uint8")
    pred_l = (pred == 1).astype("uint8")
    pred_p = (pred == 2).astype("uint8")
    pred_v = pred_l + pred_p

    gt_e = (gt == 0).astype("uint8")
    gt_l = (gt == 100).astype("uint8")
    gt_p = (gt == 255).astype("uint8")
    gt_v = gt_l + gt_p

    # area
    tot_area = prediction.shape[0] ** 2

    area_pred_l = np.count_nonzero(pred_l) / tot_area
    area_pred_p = np.count_nonzero(pred_p) / tot_area
    area_pred_v = np.count_nonzero(pred_v) / tot_area

    area_gt_l = np.count_nonzero(gt_l) / tot_area
    area_gt_p = np.count_nonzero(gt_p) / tot_area
    area_gt_v = np.count_nonzero(gt_v) / tot_area

    # error_area_l = area_pred_l - area_gt_l
    # error_area_p = area_pred_p - area_gt_p
    # error_area_v = area_pred_v - area_gt_v

    # calculando os IoUs

    # iou do externo, lumen, placa e vaso
    # (vaso = lumen + placa)
    iou_e = get_iou(pred_e, gt_e)
    iou_l = get_iou(pred_l, gt_l)
    iou_p = get_iou(pred_p, gt_p)
    iou_v = get_iou(pred_v, gt_v)

    # iou médio entre placa e lumen
    iou_avg = (iou_l + iou_p) / 2

    iou = [
        iou_avg,
        iou_e,
        iou_l,
        iou_p,
        iou_v,
        area_pred_l,
        area_gt_l,
        area_pred_p,
        area_gt_p,
        area_pred_v,
        area_gt_v,
    ]

    return iou


def retrieve_stats(
    model,
    stat_gen,
    dataset,
    verbose=1,
):
    loops = stat_gen.__len__() + 1

    # initialize the array that will hold iou information
    iou_list = [None for k in range(len(dataset))]

    for i in range(loops):

        if verbose == 1:
            print(str(i + 1) + " / " + str(loops))
        batch = stat_gen.__getitem__(i)[0]

        # if on last loop, not all batchs might be full
        if i == loops - 1:
            batch = batch[: (len(dataset) - stat_gen.batch_size * stat_gen.__len__())]
        preds = model.predict_on_batch(batch)

        # calculate base position on arrays
        base_j = i * stat_gen.batch_size

        for j in range(len(preds)):

            iou_list[j + base_j] = save_stats(preds[j], dataset.mask_path.iloc[j + base_j])
    return iou_list


def save_preds(
    model,
    stat_gen,
    data,
    save_folder,
    amount=10,
    name_format=["Average", "Name"],
    print_options=[True, True, True, True, True, True],
    verbose=1,
):
    if amount > 0:

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        loops = stat_gen.__len__() + 1

        # initialize the array that will hold iou information

        for i in range(loops):

            if verbose == 1:
                print(str(i + 1) + " / " + str(loops))
            batch = stat_gen.__getitem__(i)[0]

            # if on last loop, not all batchs might be full
            if i == loops - 1:
                batch = batch[: (len(data) - stat_gen.batch_size * stat_gen.__len__())]
            preds = model.predict_on_batch(batch)

            # calculate base position on arrays
            base_j = i * stat_gen.batch_size

            # if desired amount has not been reached, continue saving batches
            if not os.path.exists(save_folder + "/predictions"):
                os.makedirs(save_folder + "/predictions")
            for j in range(len(preds)):
                if j + base_j < amount:
                    name = pred_name(data, j, base_j, name_format=name_format)
                    save_output(
                        name,
                        preds[j],
                        data.iloc[j + base_j]["Input Image"],
                        data.iloc[j + base_j]["Ground Truth"],
                        save_folder,
                        print_options=print_options,
                    )
                else:
                    break
    return


def format_table(
    data,
    data_info,
    dataset,
    save_folder,
    sorting_metric="Average",
):
    """

    :param data:
    :param data_info:
    :param dataset:
    :param save_folder:
    :param sorting_metric:
    :return:
    """
    dataf = data.loc[:, :].astype("object")
    data_infof = data_info.loc[:, :].astype("object")

    data_infof.loc["count"] = data_infof.loc["count"].map("{:.0f}".format)

    for i in dataf.iteritems():
        dataf.loc[:, i[0]] = dataf[i[0]].map("{:.4f}".format)
        data_infof.loc["mean":"max", i[0]] = data_infof.loc["mean":"max", i[0]].map("{:.4f}".format)

    dataf = dataf.astype("str")
    data_infof = data_infof.astype("str")

    df = dataset.set_index(np.arange(len(dataset)))

    dataf.insert(0, "Name", df.file_name)

    dataf.Name = dataf.Name.apply(lambda x: x[:-4])

    dataf["Input Image"] = dataset.set_index(np.arange(len(dataset))).raw_path
    dataf["Ground Truth"] = dataset.set_index(np.arange(len(dataset))).mask_path

    dataf.to_csv(save_folder + "/metrics_pretty.csv")
    data_infof.to_csv(save_folder + "/metrics_summary_pretty.csv")

    data_pretty_sorted = dataf.sort_values(by=sorting_metric)

    # data_pretty_sorted.to_csv(save_folder + "/metrics_sorted.csv")

    return dataf, data_infof, data_pretty_sorted


def get_statistics(iou_list, save_folder, dpi=400, ci=None):
    plots_folder = save_folder + "/plots"
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)
    data = pd.DataFrame(
        np.array(iou_list, dtype="float32"),
        columns=[
            "Average",
            "Outer",
            "Lumen",
            "Plaque",
            "Vessel",
            "Lumen Area [mm²]",
            "Lumen Area GT [mm²]",
            "Plaque Area [mm²]",
            "Plaque Area GT [mm²]",
            "Vessel Area [mm²]",
            "Vessel Area GT [mm²]",
        ],
    )

    data["Lumen Area Ratio"] = data["Lumen Area [mm²]"] / data["Lumen Area GT [mm²]"]
    data["Plaque Area Ratio"] = data["Plaque Area [mm²]"] / data["Plaque Area GT [mm²]"]
    data["Vessel Area Ratio"] = data["Vessel Area [mm²]"] / data["Vessel Area GT [mm²]"]

    # Uma imagem inteira tem 100mm2
    data.loc[:, "Lumen Area [mm²]":"Vessel Area GT [mm²]"] = 100 * data.loc[:, "Lumen Area [mm²]":"Vessel Area GT [mm²]"]

    data["Plaque Burden"] = data["Plaque Area [mm²]"] / data["Vessel Area [mm²]"]
    data["Plaque Burden GT"] = data["Plaque Area GT [mm²]"] / data["Vessel Area GT [mm²]"]

    data["Plaque Burden Model/GT Ratio"] = data["Plaque Burden"] / data["Plaque Burden GT"]

    data_info = data.describe()

    data.to_csv(save_folder + "/metrics.csv")
    data_info.to_csv(save_folder + "/metrics_summary.csv")

    def make_area_ratio_plots(
        name,
        gt_name,
        ratio_name,
        file_name,
        dpi=dpi,
        data=data,
        data_info=data_info,
        ci=ci,
    ):

        sns.set_theme(style="whitegrid")

        # [min x, max x, min y, max y]
        data_max = max(data_info.loc["max", [name, gt_name]])

        graph = sns.scatterplot(x=data[gt_name], y=data[name], marker="o", color="red", s=3)
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph.get_figure().savefig(
            plots_folder + "/" + file_name + ".png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        graph.get_figure().clf()

        graph = sns.scatterplot(x=data[gt_name], y=data[name], marker="o", color="red", s=3)
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.axis("scaled")
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph.get_figure().savefig(
            plots_folder + "/" + "scaled_" + file_name + ".png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        graph.get_figure().clf()

        graph = matplotlib.pyplot.axes()
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph = sns.regplot(
            data=data,
            x=gt_name,
            y=name,
            color="red",
            truncate=False,
            ci=ci,
            scatter_kws={"s": 3},
        )
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.get_figure().savefig(
            plots_folder + "/" + "regression_" + file_name + ".png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        graph.get_figure().clf()

        graph = matplotlib.pyplot.axes()
        graph.axis("scaled")
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph = sns.regplot(
            data=data,
            x=gt_name,
            y=name,
            color="red",
            truncate=False,
            ci=ci,
            scatter_kws={"s": 3},
        )
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max])
        graph.get_figure().savefig(
            plots_folder + "/" + "regression_scaled_" + file_name + ".png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        graph.get_figure().clf()

        graph = sns.scatterplot(data=data[ratio_name], marker="o", color="red", s=3)
        graph = sns.lineplot(x=[i for i in range(len(data))], y=[1 for i in range(len(data))])
        graph.set(
            xlim=(0, len(data)),
            ylim=(0, max(2, 1.1 * data_info.loc["max", ratio_name])),
        )
        graph.get_figure().savefig(
            plots_folder + "/" + "ratio_" + file_name + ".png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        graph.get_figure().clf()

        graph = matplotlib.pyplot.axes()
        graph.set(
            xlim=(0, len(data)),
            ylim=(0, max(2, 1.1 * data_info.loc["max", ratio_name])),
        )
        graph = sns.regplot(
            data=data,
            x=[i for i in range(len(data))],
            y=data[ratio_name],
            color="red",
            truncate=False,
            ci=ci,
            scatter_kws={"s": 3},
        )
        graph = sns.lineplot(x=[i for i in range(len(data))], y=[1 for i in range(len(data))])
        graph.get_figure().savefig(
            plots_folder + "/" + "ratio_regression_" + file_name + ".png",
            format="png",
            dpi=dpi,
            bbox_inches="tight",
        )
        graph.get_figure().clf()

        return

    # no mean
    graph = sns.violinplot(data=data.loc[:, "Lumen":"Vessel"], saturation=0.9, gridsize=400, cut=0)
    graph.set(ylim=(0, 1.03))
    graph.get_figure().savefig(plots_folder + "/iou_violin.png", format="png", dpi=dpi)
    graph.get_figure().clf()

    for bw in [0.01, 0.1, 0.2]:
        graph = sns.violinplot(
            data=data.loc[:, "Lumen":"Vessel"],
            saturation=0.9,
            bw=bw,
            gridsize=400,
            cut=0,
        )
        graph.set(ylim=(0, 1.03))
        graph.get_figure().savefig(
            plots_folder + "/iou_violin_bw_" + str(bw) + ".png",
            format="png",
            dpi=dpi,
        )
        graph.get_figure().clf()
    graph = sns.boxplot(data=data.loc[:, "Lumen":"Vessel"], saturation=0.9)
    graph.set(ylim=(0, 1.03))
    graph.get_figure().savefig(plots_folder + "/iou_box.png", format="png", dpi=dpi)
    graph.get_figure().clf()

    graph = sns.scatterplot(
        data=data.loc[:, "Lumen":"Vessel"],
        markers=["o", "o", "o"],
        alpha=0.85,
        edgecolor=None,
    )
    graph.set(xlim=(0, len(data)), ylim=(0, 1.03))
    graph.get_figure().savefig(save_folder + "/plots" + "/iou_scatter.png", format="png", dpi=dpi)
    graph.get_figure().clf()

    # with mean
    idx = pd.IndexSlice
    iou_cols = idx["Lumen", "Plaque", "Vessel", "Average"]

    graph = sns.violinplot(data=data.loc[:, iou_cols], saturation=0.9, gridsize=400, cut=0)
    graph.set(ylim=(0, 1.03))
    graph.get_figure().savefig(plots_folder + "/iou_violin_avg.png", format="png", dpi=dpi)
    graph.get_figure().clf()

    for bw in [0.01, 0.1, 0.2]:
        graph = sns.violinplot(
            data=data.loc[:, iou_cols],
            saturation=0.9,
            bw=bw,
            gridsize=400,
            cut=0,
        )
        graph.set(ylim=(0, 1.03))
        graph.get_figure().savefig(
            plots_folder + "/iou_violin_bw_" + str(bw) + "_avg.png",
            format="png",
            dpi=dpi,
        )
        graph.get_figure().clf()
    graph = sns.boxplot(data=data.loc[:, iou_cols], saturation=0.9)
    graph.set(ylim=(0, 1.03))
    graph.get_figure().savefig(plots_folder + "/iou_box_avg.png", format="png", dpi=dpi)
    graph.get_figure().clf()

    graph = sns.scatterplot(data=data.loc[:, "Average"], markers=["o"], alpha=0.85, edgecolor=None)
    graph.set(ylim=(0, 1.03))
    graph.set(xlim=(0, len(data)), ylim=(0, 1.03))
    graph.get_figure().savefig(save_folder + "/plots" + "/iou_scatter_avg.png", format="png", dpi=dpi)
    graph.get_figure().clf()

    make_area_ratio_plots(
        "Lumen Area [mm²]",
        "Lumen Area GT [mm²]",
        "Lumen Area Ratio",
        "area_lumen",
    )

    make_area_ratio_plots(
        "Plaque Area [mm²]",
        "Plaque Area GT [mm²]",
        "Plaque Area Ratio",
        "area_plaque",
    )
    make_area_ratio_plots(
        "Vessel Area [mm²]",
        "Vessel Area GT [mm²]",
        "Vessel Area Ratio",
        "area_vessel",
    )

    make_area_ratio_plots(
        "Plaque Area [mm²]",
        "Vessel Area [mm²]",
        "Plaque Burden",
        "plaque_burden",
    )
    make_area_ratio_plots(
        "Plaque Area GT [mm²]",
        "Vessel Area GT [mm²]",
        "Plaque Burden GT",
        "plaque_burden_gt",
    )
    make_area_ratio_plots(
        "Plaque Burden",
        "Plaque Burden GT",
        "Plaque Burden Model/GT Ratio",
        "plaque_burden_model_gt_comparison",
    )

    return data, data_info


def plot_training(history, save_folder):
    training = pd.DataFrame()

    training["Loss"] = history.history["loss"]
    training["Val. Loss"] = history.history["val_loss"]
    training["Mean IoU"] = history.history["mean_io_u"]
    training["Val. Mean IoU"] = history.history["val_mean_io_u"]

    training["Epoch"] = np.arange(1, len(training) + 1)
    training.set_index("Epoch", inplace=True)

    sns.set_theme(style="ticks")

    graph = sns.lineplot(data=training, palette="bright", markers=True, dashes=False)
    graph.set(
        xlim=(1, len(training)),
        ylim=(0, 1),
        ylabel="Metric",
        xticks=np.arange(1, len(training) + 1),
    )
    graph.tick_params(axis="x", which="major", labelsize=7, rotation=45)
    graph.tick_params(axis="y", which="major", labelsize=8, rotation=0)
    graph.get_figure().savefig(save_folder + "/training_results.png", format="png", dpi=800)
    graph.get_figure().clf()

    training.to_csv(save_folder + "/training_results.csv")

    return training


def get_bad_preds(
    model,
    data_sorted,
    save_folder,
    image_size,
    amount=10,
    name_format=["Average", "Name"],
    verbose=1,
):
    if amount > 0:

        if not os.path.exists(save_folder + "/bad_preds"):
            os.makedirs(save_folder + "/bad_preds")
        bad_preds = data_sorted.iloc[:amount]

        idx = pd.IndexSlice

        bad_dataset = bad_preds.loc[:, idx["Input Image", "Ground Truth"]]

        bad_dataset.columns = pd.Index(data=["raw_path", "mask_path"])

        bad_gen = man.KerasManager(10, image_size, bad_dataset, normalize=True)

        save_preds(
            model=model,
            stat_gen=bad_gen,
            data=bad_preds,
            save_folder=save_folder + "/bad_preds",
            amount=amount,
            name_format=name_format,
            verbose=verbose,
        )
    return
