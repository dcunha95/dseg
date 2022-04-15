import tensorflow as tf

import pandas as pd
import seaborn as sns
import numpy as np
import os

import PIL
from PIL import ImageOps

import matplotlib


class DataUtils:
    """
    Low level data related utilities
    """

    # one at a time
    @staticmethod
    def get_path(input_dir: str) -> tuple:
        """Retrieves Tuple )"""

        input_img_paths = sorted(
            [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".png") and not fname.startswith(".")]
        )

        input_names = sorted(
            [fname.split(sep=".")[0] for fname in os.listdir(input_dir) if fname.endswith(".png") and not fname.startswith(".")]
        )

        return input_img_paths, input_names

    @staticmethod
    def build_dataset_reference(
        input_dirs,
        target_dirs,
    ) -> pd.DataFrame:
        """
        Compares files found in multiple dirs and return them in couples (if a couple is found).

        :param input_dirs:
        :param target_dirs:
        :return:
        """

        # encapsulate in a list if necessary
        if isinstance(input_dirs, str):
            input_dirs = [input_dirs]

        if isinstance(target_dirs, str):
            target_dirs = [target_dirs]

        input_paths = []
        for i in input_dirs:
            path = DataUtils.get_path(i)
            for j in range(len(path[0])):
                input_paths += [[path[1][j], path[0][j]]]
        target_paths = []
        for i in target_dirs:
            path = DataUtils.get_path(i)
            for j in range(len(path[0])):
                target_paths += [[path[1][j], path[0][j]]]
        input_paths = sorted(input_paths)
        target_paths = sorted(target_paths)

        files = []
        for i in range(len(input_paths)):
            for j in range(i, len(target_paths)):
                if input_paths[i][0] == target_paths[j][0]:
                    files += [[input_paths[i][1], target_paths[j][1], input_paths[i][0]]]
                    break
        files = pd.DataFrame(files, columns=["raw_path", "mask_path", "file_name"])
        return files

    @staticmethod
    def load_dataset_reference(
        file_path: str,
    ) -> pd.DataFrame:
        """
        Loads a .csv dataset reference file.

        :param file_path:
        :param randomize:
        :param seed:
        :return:
        """

        files = pd.read_csv(file_path)
        files.drop(columns=["Unnamed: 0"], inplace=True)

        return files

    @staticmethod
    def split_dataset(files: pd.DataFrame, split: list = [0.6, 0.2, 0.2], reset_index: bool = True):
        """
        Split a dataset according to desired split percentages.

        :param files:
        :param split:
        :return:
        """

        train_split = split[0]
        validation_split = split[1]
        test_split = split[2]

        trn_amount = int(train_split * len(files))
        val_amount = int(validation_split * len(files))
        test_amount = int(test_split * len(files))

        trn_dataset = files.iloc[-(val_amount + test_amount + trn_amount) : -(val_amount + test_amount)]
        val_dataset = files.iloc[-(val_amount + test_amount) : -test_amount]
        tst_dataset = files.iloc[-test_amount:]

        if reset_index:
            trn_dataset = trn_dataset.reset_index(drop=True)
            val_dataset = val_dataset.reset_index(drop=True)
            tst_dataset = tst_dataset.reset_index(drop=True)

        return trn_dataset, val_dataset, tst_dataset

    @staticmethod
    def shuffle_dataset(files: pd.DataFrame, seed: int = 1337) -> pd.DataFrame:
        """Shuffles dataset."""

        files = files.sample(frac=1, random_state=seed).reset_index(drop=True)

        return files

    @staticmethod
    def get_dataset_percent(files: pd.DataFrame, percent: float = 1.0, random: bool = False, seed: int = 1337) -> pd.DataFrame:
        """Retrieves fraction of dataset"""

        if random:
            files = files.sample(frac=percent, random_state=seed).reset_index(drop=True)
        else:
            files = files.iloc[: int(percent * len(files))].reset_index(drop=True)

        return files

    @staticmethod
    def update_model_name(model_name: str) -> str:
        """
        Check if model_name already exists, updates it accordingly, and finally creates folder with updated model_name.

        :param model_name:
        :return: Updated name.
        """
        k = 0
        updated_model_name = model_name + "_" + str(k)
        while os.path.exists(updated_model_name):
            k += 1
            n = [i + "_" for i in updated_model_name.split(sep="_")[:-1]]
            name = ""
            for i in n:
                name += i
            updated_model_name = name + str(k)

        os.makedirs(updated_model_name)

        return updated_model_name

    @staticmethod
    def format_table(
        data: pd.DataFrame,
        data_info: pd.DataFrame,
        dataset,
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
            data_infof.loc["mean":"IQR", i[0]] = data_infof.loc["mean":"IQR", i[0]].map("{:.4f}".format)

        data_infof.index = ['Count', "Mean", "Std", "IQR", "Min", "25%", "50%", "75%", "Max"]

        dataf = dataf.astype("str")
        data_infof = data_infof.astype("str")

        df = dataset.set_index(np.arange(len(dataset)))

        dataf.insert(0, "Name", df.file_name)

        # dataf.Name = dataf.Name.apply(
        #     lambda
        #         x: x[:-4]
        #     )

        dataf["Input Image"] = dataset.set_index(np.arange(len(dataset))).raw_path
        dataf["Ground Truth"] = dataset.set_index(np.arange(len(dataset))).mask_path

        data_sorted = dataf.sort_values(by=sorting_metric)

        # data_sorted.to_csv(save_folder + "/metrics_sorted.csv")

        return dataf, data_infof, data_sorted


class TrainingUtils:
    """
    Contains low level training related utilities.
    """

    @staticmethod
    def iou(y_pred, y_true) -> float:
        """Returns IoU (Jaccard Index)"""
        inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
        union = np.count_nonzero(np.logical_or(y_pred, y_true).astype("uint8"))
        return inter / union

    @staticmethod
    def dice(y_pred, y_true) -> float:
        """Returns DICE (Sørensen–Dice Index)"""
        inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
        x = np.count_nonzero(y_pred.astype("uint8"))
        y = np.count_nonzero(y_true.astype("uint8"))
        return inter / (x + y)

    @staticmethod
    @tf.function
    def iou_loss(y_true, y_pred) -> float:
        """Returns IoU loss."""
        inter = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[1, 2])
        union = tf.math.reduce_sum(y_true + y_pred, axis=[1, 2]) - inter
        iou_loss = 1 - tf.math.reduce_mean(tf.math.divide_no_nan(inter, union), axis=1)

        return iou_loss

    @staticmethod
    def iou_squared_loss(y_true, y_pred) -> float:
        """Returns Squared IoU loss."""
        inter = tf.math.reduce_sum(tf.math.pow(tf.math.multiply(y_pred, y_true), 2), axis=[1, 2])
        union = tf.math.reduce_sum(y_true + y_pred, axis=[1, 2]) - inter
        iou_loss = 1 - tf.math.reduce_mean(tf.math.divide_no_nan(inter, union), axis=1)

        return iou_loss

    @staticmethod
    def get_scheduler_function(scheduler_type: str = "exp_decay", threshold=9, decay=0.1):
        def scheduler(epoch, lr):
            if epoch < threshold:
                return lr
            else:
                return lr * tf.math.exp(-decay)

        return scheduler

    @staticmethod
    @tf.function
    def prep_x(file_path, image_size=(512, 512)):
        """Default preprocessing routine for inputs, from file path to input tensor ready for training."""

        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=image_size)
        return img

    @staticmethod
    @tf.function
    def prep_y(file_path, image_size=(512, 512)):
        """Default preprocessing routine for ground truths, from file path to input tensor ready for training."""

        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, size=image_size, method="nearest")
        img = tf.stack([img == 0, img == 100, img == 255], axis=3)
        img = tf.reshape(img, shape=(image_size[0], image_size[1], 3))
        img = tf.cast(img, dtype=tf.float32)
        return img

    @staticmethod
    @tf.function
    def split_y(file_path, image_size=(512, 512)):
        """Preprocessing routine that splits the mask in two separate ground truths (lumen and vessel)."""
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, size=image_size, method="nearest")

        lumen = tf.cast(img == 100, dtype=tf.float32) * 1
        vessel = tf.cast(img == 100, dtype=tf.float32) * 1 + tf.cast(img == 255, dtype=tf.float32) * 1

        return lumen, vessel

    @staticmethod
    @tf.function
    def join_y(lumen, vessel):
        """Postprocessing routine that joins previously split mask."""

        # print(lumen.shape, vessel.shape)
        # print(lumen.shape[1:-1])
        lumen = tf.reshape(lumen, shape=lumen.shape[1:-1])
        vessel = tf.reshape(vessel, shape=vessel.shape[1:-1])

        # outer equals not vessel
        outer = 1 - vessel

        # plaque equals vessel minus lumen
        plaque = vessel - lumen

        y = tf.stack([outer, lumen, plaque], axis=-1)

        return y

    @staticmethod
    def get_tf_dataset(
        ds: pd.DataFrame,
        image_size: tuple = (512, 512),
        batch_size: int = 1,
        shard: bool = True,
        prep_x=None,
        prep_y=None,
        return_y: bool = True,
    ):
        """
        Returns a tf.data dataset instance from a DataFrame object, with preprocessing and other data manipulation routines applied.

        :param ds:
        :param image_size:
        :param batch_size:
        :param shard:
        :param prep_x:
        :param prep_y:
        :param return_y:
        :return:
        """

        if prep_x is None:
            prep_x = TrainingUtils.prep_x

        if return_y:
            if prep_y is None:
                prep_y = TrainingUtils.prep_y

            def prep_ds(x, y):
                px = prep_x(x, image_size=image_size)
                py = prep_y(y, image_size=image_size)
                return px, py

            tf_ds = tf.data.Dataset.from_tensor_slices((ds.raw_path, ds.mask_path))

        else:

            def prep_ds(x):
                px = prep_x(x, image_size=image_size)
                return px

            tf_ds = tf.data.Dataset.from_tensor_slices(ds.raw_path)

        options = tf.data.Options()

        if shard:
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        else:
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF

        tf_ds = tf_ds.with_options(options)

        tf_ds = tf_ds.map(
            prep_ds,
            # num_parallel_calls=tf.data.AUTOTUNE,
        )
        tf_ds = tf_ds.batch(batch_size, drop_remainder=True)
        tf_ds = tf_ds.prefetch(tf.data.AUTOTUNE)

        return tf_ds

    @staticmethod
    def prediction_metrics(prediction, target_img_path):
        """
        Takes quality metrics from a prediction.

        :param prediction:
        :param target_img_path:
        :return:
        """

        pred = np.argmax(prediction, axis=-1)

        gt = tf.keras.preprocessing.image.load_img(
            path=target_img_path,
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
        tot_area = prediction.shape[1] ** 2

        area_pred_l = np.count_nonzero(pred_l) / tot_area
        area_pred_p = np.count_nonzero(pred_p) / tot_area
        area_pred_v = np.count_nonzero(pred_v) / tot_area

        area_gt_l = np.count_nonzero(gt_l) / tot_area
        area_gt_p = np.count_nonzero(gt_p) / tot_area
        area_gt_v = np.count_nonzero(gt_v) / tot_area

        # Calculating IoUs

        # external, lumen, plaque and vessel
        # (vessel = lumen + placa)
        iou_e = TrainingUtils.iou(pred_e, gt_e)
        iou_l = TrainingUtils.iou(pred_l, gt_l)
        iou_p = TrainingUtils.iou(pred_p, gt_p)
        iou_v = TrainingUtils.iou(pred_v, gt_v)

        # average iou between plaque and lumen
        iou_avg = (iou_l + iou_p) / 2

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


class PlotUtils:
    """
    Contains low level plotting related utilities
    """

    @staticmethod
    def pred_name(data: pd.DataFrame, j, base_j, name_format=["Average", "Name"]):
        """Returns a prediction's file name."""
        name = ""

        for i in name_format:
            if i in data.columns:
                name += str(data.iloc[j + base_j][i]) + "_"

        name += str(data.index[base_j + j])
        name = name.replace(".", "")

        return name

    @staticmethod
    def save_output(
        name: str,
        pred,
        save_folder: str,
        input_img_path: str = "",
        target_img_path: str = "",
        print_options=[True, True, True, True, True, True],
    ):
        """Saves output"""

        print_options = print_options

        # if no input path is passed, do not try to generate ground truth images
        if input_img_path == "":
            print_options[2] = False
            print_options[3] = False

        # if no target path is passed, do not try to generate ground truth images
        if target_img_path == "":
            print_options[4] = False
            print_options[5] = False

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
        Save Ratio Plots

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
        data_max += 0.0001

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

        PlotUtils.ratio_plots(
            name="Lumen Area [mm²]",
            gt_name="Lumen Area GT [mm²]",
            ratio_name="Lumen Area Ratio",
            file_name="area_lumen",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        PlotUtils.ratio_plots(
            name="Plaque Area [mm²]",
            gt_name="Plaque Area GT [mm²]",
            ratio_name="Plaque Area Ratio",
            file_name="area_plaque",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        PlotUtils.ratio_plots(
            name="Vessel Area [mm²]",
            gt_name="Vessel Area GT [mm²]",
            ratio_name="Vessel Area Ratio",
            file_name="area_vessel",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )

        PlotUtils.ratio_plots(
            name="Plaque Area [mm²]",
            gt_name="Vessel Area [mm²]",
            ratio_name="Plaque Burden",
            file_name="plaque_burden",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        PlotUtils.ratio_plots(
            name="Plaque Area GT [mm²]",
            gt_name="Vessel Area GT [mm²]",
            ratio_name="Plaque Burden GT",
            file_name="plaque_burden_gt",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )
        PlotUtils.ratio_plots(
            name="Plaque Burden",
            gt_name="Plaque Burden GT",
            ratio_name="Plaque Burden Model/GT Ratio",
            file_name="plaque_burden_model_gt_comparison",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )

        return
