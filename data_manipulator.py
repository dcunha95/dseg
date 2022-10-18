import os

import tensorflow as tf

import pandas as pd
import seaborn as sns
import numpy as np
import skimage as ski 
from scipy.spatial.distance import directed_hausdorff
from itertools import product

import PIL
from PIL import ImageOps

import matplotlib
import matplotlib.pyplot as plt


class DataUtils:
    """
    Low level data related utilities
    """

    @staticmethod
    def make_path(path : str = ""):
        """Checks if path exists. If not, creates it."""
        
        try:
            if not os.path.exists(path):
                os.makedirs(path) 
        except Exception as exception:
            raise(exception)

    @staticmethod
    def simple_counter(i, total):
        """Prints simple counter """

        flourish= ['\\', '|', '/', '-']

        s_i = str(i+1)
        s_total = str(total)
        while len(s_i) < len(s_total):
            s_i = "".join(["0", s_i])

        if i+1 == total:
            print("".join(["\r", s_i, " / ", s_total]), end='\n')
        else:
            print("".join(["\r", s_i, " / ", s_total, '  ', flourish[(i+1) % 4]]), end='')

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
    def process_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
        """Adds more columns to dataset."""

        get_artery_info = lambda x: ''.join([i+'_' for i in x.split('_')[:-1]])[:-1]
        get_frame_number = lambda x: int(x.split('_')[-1])

        dataset["artery_info"] = [get_artery_info(j) for (i,j) in dataset.file_name.iteritems()]
        dataset["frame_number"] = [get_frame_number(j) for (i,j) in dataset.file_name.iteritems()]
            
        return dataset.copy()

    @staticmethod
    def prune_dataset(ds, channels, strides) -> pd.DataFrame:
        """Prepares a dataset for use with multichannels."""

        border = int(strides*(channels-1)/2)

        # get processed data
        arteries = ds.artery_info.unique()

        # prune artery borders
        df = pd.DataFrame()
        for artery_i in arteries:
            df_i = ds.loc[ds.artery_info == artery_i]
            
            df_i.sort_values("frame_number", inplace=True)
            df_i.reset_index(drop=True, inplace=True)

            df_i = df_i.iloc[border:-border]
            df = pd.concat([df, df_i])

        df.reset_index(drop=True, inplace=True)

        return df.copy()

    @staticmethod
    def get_available_from_dataset(dataset, available_dataset) -> pd.DataFrame:
        """Returns items from dataset found in available_dataset."""

        df = dataset.loc[[j in available_dataset.file_name.to_list() for (i,j) in dataset.file_name.iteritems()]].copy()
        df.reset_index(drop=True, inplace=True)

        return df.copy()

    @staticmethod
    def get_dataset_percent(files: pd.DataFrame, percent: float = 1.0, random: bool = False, seed: int = 1337) -> pd.DataFrame:
        """Retrieves fraction of dataset"""

        if random:
            files = files.sample(frac=percent, random_state=seed).reset_index(drop=True)
        else:
            files = files.iloc[: int(percent * len(files))].reset_index(drop=True)

        return files


    @staticmethod
    def get_multichannels(file_path: str, channels: int = 3, strides: int = 1) -> list:
        """Return list of files for 2.5D training."""


        print(file_path)
        file_segs = tf.strings.split(file_path, sep="/")
        base = tf.strings.join(file_segs[:-1], separator="/")
        name = tf.unstack(tf.strings.split(file_segs[-1], sep="."))[0]

        frame_info = tf.strings.split(name, sep="_")

        frame_number = tf.strings.to_number(frame_info[-1], out_type=tf.dtypes.int32)

        # base path + artery info
        base = tf.strings.join([base, tf.strings.join(frame_info[:-1], separator="_")], separator="/")

        side_channels = int((channels-1)/2)

        # all frames
        frame_list = tf.range(-side_channels*strides+frame_number, side_channels*strides+1+frame_number, strides)
        frame_list = tf.strings.as_string(frame_list)

        frame_path_list = tf.map_fn(
            lambda x: tf.strings.join([base, "_", x,".png"], separator=""),
            frame_list,
        )

        return frame_path_list


    @staticmethod
    def _get_multichannels(file_path: str, channels: int = 3, strides: int = 1) -> list:
        """Return list of files for 2.5D training."""


        # print(file_path)
        base, name = os.path.split(file_path)
        # name = file_path.split('/')[-1]
        # base = file_path[:-(len(name)+1)]

        frame_info = name[:-4].split('_')


        # base path + artery info
        base = os.path.join(base, ''.join([i+'_' for i in frame_info[:-1]]))


        frame_number = int(frame_info[-1])

        # deltas for retrieving support frames
        side_frames = int((channels-1)/2) 
        frame_dif_list = [i for i in range(-side_frames*strides, side_frames*strides+1, strides)]

        frame_path_list = []

        # base string construction
        l = [base, 0, '.png']
        for frame_dif in frame_dif_list:
            # support frame number
            l[1] = str(frame_number + frame_dif)
            
            # build complete path string and append to list 
            frame_path_list.append(''.join(i for i in l))

        return tuple(frame_path_list)

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
        sorting_metric=("IoU", "Average"),
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

        data_infof.index = ["Count", "Mean", "Std", "IQR", "Min", "25%", "50%", "75%", "Max"]

        dataf = dataf.astype("str")
        data_infof = data_infof.astype("str")

        df = dataset.set_index(np.arange(len(dataset)))

        # dataf.insert(0, "Name", df.file_name)

        # dataf.Name = dataf.Name.apply(
        #     lambda
        #         x: x[:-4]
        #     )

        dataf[("Path", "File Name")] = dataset.set_index(np.arange(len(dataset))).file_name
        dataf[("Path", "Input Image")] = dataset.set_index(np.arange(len(dataset))).raw_path
        dataf[("Path", "Ground Truth")] = dataset.set_index(np.arange(len(dataset))).mask_path
        
        data_sorted = dataf.sort_values(by=sorting_metric)

        # data_sorted.to_csv(save_folder + "/metrics_sorted.csv")

        return dataf, data_infof, data_sorted

    @staticmethod
    def make_tables(analysis, path="", drop_count=False):
        """Generate appropriate LaTeX files at path/latex/."""

        save_folder = os.path.join(path, "latex")

        DataUtils.make_path(save_folder)

        metrics = {
            'iou': [*product(["IoU"], ["Average", "Lumen", "Plaque", "Vessel",])],
            'dice': [*product(["DICE"], ["Average", "Lumen", "Plaque", "Vessel",])],
            'hd': [*product(["Hausdorff Distance [mm]"], ["Lumen", "Plaque", "Vessel",])],
            'area': [*product(["Area [mm²]"], ["Lumen", "Lumen GT", "Plaque", "Plaque GT", "Vessel", "Vessel GT",])],
            'ratio': [*product(["Area Ratio"], ["Lumen", "Plaque", "Vessel",])],
            'pb': [*product(["Plaque Burden"], ["Prediction", "Ground Truth", "Ratio",])],
        }

        if drop_count:
            data_info = analysis['data_info_formatted'].drop('Count')
        else:
            data_info = analysis['data_info_formatted']

        for metric in metrics:
            data_info[metrics[metric]].to_latex(os.path.join(save_folder, f'results_{metric}.tex'))
        
        data_info.to_latex(os.path.join(save_folder, 'results_all.tex'))
        data_info.T.to_latex(os.path.join(save_folder, 'results_all_t.tex'))



    @staticmethod
    def print_options_to_array(print_options):
        """Transforms a list of print options to optimized array"""

        all_options = ["raw", "output", "input", "input_original", "gt", "gt_original", "channels", "contour", "3d"]
        return [option_i in print_options for option_i in all_options]

    @staticmethod
    def divide_no_nan(a, b):
        """Calculates a/b and returns NaNs as zeros, also returning inf or -inf if b == 0"""

        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.divide(a, b)

        if np.isnan(c):
            return 0
        else:
            return c






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
    def hausdorf_distance(u, v) -> float:
        """Returns Hausdorff distance"""

        return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])

    @staticmethod
    def dice(y_pred, y_true) -> float:
        """Returns DICE (Sørensen-Dice Index)"""
        # inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
        # x = np.count_nonzero(y_pred.astype("uint8"))
        # y = np.count_nonzero(y_true.astype("uint8"))
        # return inter / (x + y)
        inter = np.count_nonzero(np.logical_and(y_pred, y_true).astype("uint8"))
        union = np.count_nonzero(np.logical_or(y_pred, y_true).astype("uint8"))
        return 2*inter / (union+inter)


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
    def prep_x(file_path, image_size=(512, 512), **kwargs):
        """Default preprocessing routine for inputs, from file path to input tensor ready for training."""

        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, size=image_size)
        return img


    @staticmethod
    def multi_x(file_path: tuple, channels: int = 3, channel_strides: int = 1, image_size=(512, 512), **kwargs):
        """Multichannel loading routine."""

        # file_path = file_path.numpy().decode('UTF-8')
        # file_path = np.array(file_path).item().decode("UTF-8")

        # files_list = DataUtils.get_multichannels(file_path, channels, channel_strides)

        # img = tf.stack([TrainingUtils.prep_x(i, image_size=image_size) for i in file_path], axis=-1)

        img = tf.map_fn(
            # lambda i: TrainingUtils.prep_x(i, image_size=image_size),
            lambda i: tf.reshape(TrainingUtils.prep_x(i, image_size=image_size), image_size),
            file_path,
            fn_output_signature=tf.float32,
        )
        
        img = tf.stack(tf.unstack(img, axis=0), axis=2)

        #nasty bug...
        # img = tf.reshape(img, shape=(*image_size, channels))

        #can't have loops in graphs...
        # img = tf.stack([i for i in img], axis=2)
        # img = tf.reshape(img, shape=(*image_size, channels))

        # shape = tf.ensure_shape(img, [None, image_size[0], image_size[1], channels]        
        
        return img

    @staticmethod
    @tf.function
    def prep_y(file_path, image_size=(512, 512), **kwargs):
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
    def split_y(file_path, image_size=(512, 512), **kwargs):
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
        channels: int = 1,
        strides: int = 1,
        shard: bool = True,
        prep_x=None,
        prep_y=None,
        return_y: bool = True,
        **kwargs,
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
            # if doing multichannel, use multi_x prep
            if channels == 1:
                prep_x = TrainingUtils.prep_x
            else:
                prep_x = TrainingUtils.multi_x

        # if doing multichannel, tf.data's "x" will be a list of files paths instead of a single string
        if channels != 1:
            raw_paths = [DataUtils._get_multichannels(j, channels, strides) for (i,j) in ds.raw_path.iteritems()]

        if return_y:
            if prep_y is None:
                prep_y = TrainingUtils.prep_y

            @tf.function
            def prep_ds(x, y):
                px = prep_x(x, image_size=image_size, channels=channels, channel_strides=strides)
                py = prep_y(y, image_size=image_size, channels=channels, channel_strides=strides)
                # shape_x = tf.ensure_shape(px, [None, image_size[0], image_size[1], channels])
                return px, py

            if channels != 1:
                tf_ds = tf.data.Dataset.from_tensor_slices((raw_paths, ds.mask_path))
            else:
                tf_ds = tf.data.Dataset.from_tensor_slices((ds.raw_path, ds.mask_path))

        else:

            def prep_ds(x):
                px = prep_x(x, image_size=image_size, channels=channels, channel_strides=strides)
                # shape_x = tf.ensure_shape(px, [None, image_size[0], image_size[1], channels])
                return px

            if channels != 1:
                tf_ds = tf.data.Dataset.from_tensor_slices(raw_paths)
            else:
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


        # Calculating Areas (100mm^2 * class_pixels/total_pixels)

        tot_area = prediction.shape[1] ** 2

        area_pred_l = 100*np.count_nonzero(pred_l) / tot_area
        area_pred_p = 100*np.count_nonzero(pred_p) / tot_area
        area_pred_v = 100*np.count_nonzero(pred_v) / tot_area

        area_gt_l = 100*np.count_nonzero(gt_l) / tot_area
        area_gt_p = 100*np.count_nonzero(gt_p) / tot_area
        area_gt_v = 100*np.count_nonzero(gt_v) / tot_area


        # Calculating IoUs

        # external, lumen, plaque and vessel
        # (vessel = lumen + placa)
        iou_e = TrainingUtils.iou(pred_e, gt_e)
        iou_l = TrainingUtils.iou(pred_l, gt_l)
        iou_p = TrainingUtils.iou(pred_p, gt_p)
        iou_v = TrainingUtils.iou(pred_v, gt_v)

        # average iou between plaque and lumen
        iou_avg = (iou_l + iou_p) / 2


        # Calculating DICEs

        # external, lumen, plaque and vessel
        # (vessel = lumen + plaque)
        dice_e = TrainingUtils.dice(pred_e, gt_e)
        dice_l = TrainingUtils.dice(pred_l, gt_l)
        dice_p = TrainingUtils.dice(pred_p, gt_p)
        dice_v = TrainingUtils.dice(pred_v, gt_v)

        # average dice between plaque and lumen
        dice_avg = (dice_l + dice_p) / 2


        # Calculating Hausdorff Distances

        # lumen hausdorf distance
        pred_l_contours = PlotUtils._get_contours(pred_l)
        gt_l_contours = PlotUtils._get_contours(gt_l)

        if len(pred_l_contours) != 0 and len(gt_l_contours) != 0:

            pred_l_contour = pred_l_contours[0]
            gt_l_contour = gt_l_contours[0]

            hd_l = TrainingUtils.hausdorf_distance(pred_l_contour, gt_l_contour)

        else:
            # hd_l = np.inf
            hd_l = 100

        # vessel hausdorf distance
        pred_v_contours = PlotUtils._get_contours(pred_v)
        gt_v_contours = PlotUtils._get_contours(gt_v)

        if len(pred_v_contours) != 0 and len(gt_v_contours) != 0:

            pred_v_contour = pred_v_contours[0]
            gt_v_contour = gt_v_contours[0]
        
            hd_v = TrainingUtils.hausdorf_distance(pred_v_contour, gt_v_contour)
        
        else:
            # hd_v = np.inf
            hd_v = 100

        # convert to mm
        hd_l = hd_l*10/prediction.shape[1]
        hd_v = hd_v*10/prediction.shape[1]

        # plaque hausdorf distance
        hd_p = max(hd_l, hd_v)


        # Calculating Area Ratios

        ratio_l = DataUtils.divide_no_nan(area_pred_l, area_gt_l)
        ratio_p = DataUtils.divide_no_nan(area_pred_p, area_gt_p)
        ratio_v = DataUtils.divide_no_nan(area_pred_v, area_gt_v)


        # Calculating Plaque Burden

        pb_pred = DataUtils.divide_no_nan(area_pred_p, area_pred_v)
        pb_gt = DataUtils.divide_no_nan(area_gt_p, area_gt_v)
        pb_ratio = DataUtils.divide_no_nan(pb_pred, pb_gt)


        metrics = [
            # iou
            iou_avg,
            iou_e,
            iou_l,
            iou_p,
            iou_v,
            
            # dice
            dice_avg,
            dice_e,
            dice_l,
            dice_p,
            dice_v,
            
            # hausdorf
            hd_l,
            hd_p,
            hd_v,

            # area
            area_pred_l,
            area_gt_l,
            area_pred_p,
            area_gt_p,
            area_pred_v,
            area_gt_v,

            # area ratio (prediction / ground truth)
            ratio_l,
            ratio_p,
            ratio_v,

            # plaque burden
            pb_pred, 
            pb_gt,
            pb_ratio,
        ]

        return metrics


class PlotUtils:
    """
    Contains low level plotting related utilities
    """

    @staticmethod
    def pred_name(data: pd.DataFrame, j, base_j, name_format=["Average", "Name"]):
        """Returns a prediction's file name."""
        
        name_to_col = {
            "Average": ("IoU", "Average"),
            "Lumen": ("IoU", "Lumen"),
            "Plaque": ("IoU", "Plaque"),
            "Vessel": ("IoU", "Lumen"),            
            "Name": ('Path', 'File Name'),
        }

        name_to_col2 = {
            "Name": "file_name",
        }
        
        name = ""

        for i in name_format:
            i_col = name_to_col[i]
            if i_col in data.columns:
                name += str(data.iloc[j + base_j][i_col]) + "_"
            
            elif i in name_to_col2:
                i_col = name_to_col2[i]
                if i_col in data.columns:
                    name += str(data.iloc[j + base_j][i_col]) + "_"
            

        name += str(data.index[base_j + j])
        name = name.replace(".", "")

        return name

    @staticmethod
    def _get_output_save_path(base_path : str, option : str, name:str, kname : str = ""):
        """Get standard save path for an output"""

        # base_path = save_folder/predictions
        if kname != "":
            # return save_folder/predictions/option/name_kname_option.png
            return "".join([os.path.join(base_path, option, name), "_", kname, "_", option, ".png"])
        else:
            # return save_folder/predictions/option/name_option.png
            return "".join([os.path.join(base_path, option, name), "_", option, ".png"])

    @staticmethod
    def _get_contours(array, level=None):
        """Return contours"""

        return ski.measure.find_contours(array, level=level)
        
    @staticmethod
    def _paint(array, coordinates, color=[255,0,0], width=1):

        x = round(coordinates[0])
        y = round(coordinates[1])

        array[x-width:x+width+1, y-width:y+width+1, :] = color


    @staticmethod
    def save_output(
        name: str,
        pred,
        save_folder: str,
        input_img_path: str = "",
        target_img_path: str = "",
        print_options: list = [True, True, True, True, True, True, True, True, True],
        image_size=(512, 512),
        x = None,
        channels=1,
    ):
        """Saves output"""

        base_path = os.path.join(save_folder, "predictions")

        # ["raw", "output", "input", "input-original", "gt", "gt-original", "channels", "contour"]
        if isinstance(print_options[0], str):
            # make appropriate save_folder/print_options[i] paths if they don't already exist
            for option in print_options:
                path = os.path.join(base_path, option)
                DataUtils.make_path(path)

            print_options = DataUtils.print_options_to_array(print_options)


        if print_options[0]:
            img = PIL.ImageOps.autocontrast(tf.keras.preprocessing.image.array_to_img(pred))
            img.save(PlotUtils._get_output_save_path(base_path, "raw", name), format="png")
        
        if print_options[1]:
            img = np.array(np.argmax(pred, axis=-1), dtype="uint8")
            img = np.expand_dims((img == 1).astype("uint8") * 100 + (img == 2).astype("uint8") * 255, axis=-1)
            img = tf.keras.preprocessing.image.array_to_img(img, scale=False)
            img.save(PlotUtils._get_output_save_path(base_path, "output", name), format="png")
        
        if input_img_path != "":
            if print_options[2]:
                img = tf.keras.preprocessing.image.load_img(input_img_path, color_mode="grayscale", target_size=image_size, interpolation="nearest")
                img.save(PlotUtils._get_output_save_path(base_path, "input", name), format="png")
        
            if print_options[3]:
                img = tf.keras.preprocessing.image.load_img(input_img_path)
                img.save(PlotUtils._get_output_save_path(base_path, "input-original", name), format="png")


        if target_img_path != "":
            if print_options[4]:
                img = tf.keras.preprocessing.image.load_img(target_img_path, color_mode="grayscale", target_size=image_size, interpolation="nearest")
                img.save(PlotUtils._get_output_save_path(base_path, "gt", name), format="png")
            
            if print_options[5]:
                img = tf.keras.preprocessing.image.load_img(target_img_path)
                img.save(PlotUtils._get_output_save_path(base_path, "gt-original", name), format="png")

        if print_options[6] and x is not None:
            multi = tf.unstack(tf.unstack(x)[0], axis=-1)
            multi_img = [tf.keras.preprocessing.image.array_to_img(np.expand_dims(frame_i, axis=-1)) for frame_i in multi]
            for (i, img_i) in enumerate(multi_img):
                si = str(i)
                while len(si) < 3:
                    si = "0" + si

                img_i.save(PlotUtils._get_output_save_path(base_path, "channels", name, si), format="png")

        if print_options[7] and input_img_path != "":

            img = tf.keras.preprocessing.image.load_img(input_img_path, target_size=image_size, interpolation="nearest")
            img = tf.keras.preprocessing.image.img_to_array(img)

            # orange = [255, 150, 0]
            # blue = [0, 150, 255]

            if target_img_path != "":
                # ground-truth contour
                gt_array = tf.keras.preprocessing.image.load_img(target_img_path, color_mode="grayscale", target_size=image_size, interpolation="nearest")
                gt_array = tf.keras.preprocessing.image.img_to_array(gt_array)
                gt_array = np.squeeze(gt_array)
                gt_contours = PlotUtils._get_contours(gt_array)
                for i in gt_contours:
                    for j in i:
                        PlotUtils._paint(img, j, [255, 150, 0], width=1)
                        img[round(j[0]), round(j[1]), :] = [255, 150, 0]


            # predictions contour            
            pred_array = np.array(np.argmax(pred, axis=-1), dtype="uint8")
            pred_array = (pred_array == 1).astype("uint8") * 100 + (pred_array == 2).astype("uint8") * 255
            pred_contours = PlotUtils._get_contours(pred_array)
            for i in pred_contours:
                for j in i:
                    PlotUtils._paint(img, j, [0, 150, 255], width=1)
                    
            img = tf.keras.preprocessing.image.array_to_img(img)
            img.save(PlotUtils._get_output_save_path(base_path, "contour", name), format="png")

        # if channels != 1 and print_options[8]:
        if print_options[8]:
            img = tf.stack([x[:, :, :, 0], x[:, :, :, int((channels-1)/2)], x[:, :, :, -1]], axis=-1)
            img = tf.keras.preprocessing.image.array_to_img(img[0])

            img.save(PlotUtils._get_output_save_path(base_path, "3d", name), format="png")


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
    def save_plots_old(
        data,
        data_info,
        save_folder,
        dpi=400,
        ci=None,
    ):
        """
        Makes a bunch of plots at save_folder/plots. Probably not working.

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
            ratio_name="PB. Ratio",
            file_name="plaque_burden_model_gt_comparison",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )

        return
    
    @staticmethod
    def save_plots2(
        data,
        data_info,
        save_folder,
        dpi=400,
        ci=None,
    ):
        """
        Makes a bunch of plots at save_folder/plots.

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

        
        idx = pd.IndexSlice

        # IoU



        # DICE

        # Hausdorff Distance

        # Area

        # Area Ratio

        # Plaque Burden

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
            ratio_name="PB. Ratio",
            file_name="plaque_burden_model_gt_comparison",
            data=data,
            data_info=data_info,
            plots_folder=plots_folder,
        )

        return
    
    @staticmethod
    def standard_plot_routine(data: pd.DataFrame, save_folder: str, plotter = None, kname: str = '', auto_close: bool = True):
        """Standard plotting routine, saving generated plots at save_folder/plots_kname"""

        plots_folder = os.path.join(save_folder, 'plots')
        if kname != '':
            plots_folder = ''.join([plots_folder, '_', kname])
        DataUtils.make_path(plots_folder)

        if plotter is None:
            dpi = 250
            figure_args = {
                "figsize": (4*19.20/5, 4*10.80/5),
                "dpi": dpi,
            }
            plotter = GraphMaker(figure_args=figure_args)
        else:
            if 'dpi' in plotter.figure_args:
                dpi = plotter.figure_args['dpi']
            else:
                dpi = 250

        # overall scatter, violin and box plots
        for metric_i in ["DICE", "IoU", "Hausdorff Distance [mm]"]:
            metric_name = metric_i.lower().replace(' [mm]', '').replace(' ', '_')

            graph = plotter.violin(data, metric_i)
            graph.get_figure().savefig(os.path.join(plots_folder, f'violin_{metric_name}.png'), dpi=dpi, bbox_inches='tight')
            if auto_close:
                plt.close()

            graph = plotter.scatter(data, metric_i)
            graph.get_figure().savefig(os.path.join(plots_folder, f'scatter_{metric_name}.png'), dpi=dpi, bbox_inches='tight')
            if auto_close:
                plt.close()

            # scatter plot ordered by ground truth's plaque burden
            graph = plotter.scatter(data=data, metric=metric_i, x_axis=('Plaque Burden', 'Ground Truth'))
            graph.get_figure().savefig(os.path.join(plots_folder, f'scatter_x-pb_gt_y-{metric_name}.png'), dpi=dpi, bbox_inches='tight')
            if auto_close:
                plt.close()

            # scatter plot ordered by ground truth's plaque area
            graph = plotter.scatter(data=data, metric=metric_i, x_axis=('Area [mm²]', 'Plaque GT'))
            graph.get_figure().savefig(os.path.join(plots_folder, f'scatter_x-area_p_gt_y-{metric_name}.png'), dpi=dpi, bbox_inches='tight')
            if auto_close:
                plt.close()

            if metric_i in ["DICE", "IoU"]:
                graph = plotter.scatter(data, metric_i, use_average=True)
                graph.get_figure().savefig(os.path.join(plots_folder, f'scatter_{metric_name}_avg.png'), dpi=dpi, bbox_inches='tight')
                if auto_close:
                    plt.close()

                # scatter plot ordered by ground truth's plaque burden
                graph = plotter.scatter(data=data, metric=metric_i, x_axis=('Plaque Burden', 'Ground Truth'), use_average=True)
                graph.get_figure().savefig(os.path.join(plots_folder, f'scatter_x-pb_gt_y-{metric_name}_avg.png'), dpi=dpi, bbox_inches='tight')
                if auto_close:
                    plt.close()

                # scatter plot ordered by ground truth's plaque area
                graph = plotter.scatter(data=data, metric=metric_i, x_axis=('Area [mm²]', 'Plaque GT'), use_average=True)
                graph.get_figure().savefig(os.path.join(plots_folder, f'scatter_x-area_p_gt_y-{metric_name}_avg.png'), dpi=dpi, bbox_inches='tight')
                if auto_close:
                    plt.close()


        # ratio plots
        for metric_i in ["Area [mm²]", "Plaque Burden"]:
            pass
        




class GraphMaker:
    """Contains templates for plotting figures (violin, box, etc...)"""

    def __init__(self, set_title=True, palette="bright", figure_args=None, metric_translation = None):
        
        self.idx = pd.IndexSlice

        self.set_title = set_title
        self.palette = palette
        
        if figure_args is None:
            self.figure_args = {}
        else:
            self.figure_args = figure_args.copy()

        if metric_translation is None:
            self.metric_translation = {}
        else:
            self.metric_translation = metric_translation

        # sns.set_palette(self.palette)

        
    def _format_df(self, df, x=None): 
        """Formats appropriately for sns.plot(x='x', y='y', data=data) style plots"""

        df2 = pd.DataFrame()

        for (i, j) in df.iteritems():
            df_i = df.loc[:, [i]]
            df_i.columns = ['Value']
            df_i['Metric'] = i[0]
            df_i['Class'] = i[1]
            
            # add x-axis
            if x is not None:
                df_i['x'] = df.loc[:, [x]]

            df2 = pd.concat([df2, df_i])
        
        df2['data_point'] = df2.index.to_list()
        df2.reset_index(inplace=True, drop=True)

        return df2


    def _translate_metric(self, metric):
        """Translates metric's name to a fancy pantsy one (ex. 'IoU' -> 'Jaccard Index')"""
        
        if metric in self.metric_translation:
            return self.metric_translation[metric]
        else:
            return metric


    def _set_ylim(self, graph, metric):
        """Set custom limits"""

        if metric in ["DICE", "IoU"]:
            graph.set(ylim=(0, 1.03))
        
        else:
            graph.set(
                # ylim=(graph.get_ylim()[0]*1.5, graph.get_ylim()[1]*2.5),
                ylim=(0, 12),
            )


    def violin(self, data, metric, bw=0.2):
        """Typical Violin plot"""

        df = self._format_df(data)
        df = df.loc[df["Metric"] == metric]
        df = df.loc[df['Class'] != "Outer"]

        plt.figure(**self.figure_args)
        graph = sns.violinplot(data=df, x='Class', y='Value', bw=bw, saturation=0.9, gridsize=400, cut=0, palette=self.palette)
        
        self._set_ylim(graph, metric)
        graph.set(ylabel=self._translate_metric(metric))
        graph.tick_params(left=True, bottom=False)
        sns.despine(top=True, left=False, right=True, bottom=True, trim=True)
        
        return graph

        
    def scatter(self, data, metric, use_average=False, x_axis=None):
        """Typical Scatter plot"""

        df = self._format_df(data, x=x_axis)
        df = df.loc[df["Metric"] == metric]
        
        plt.figure(**self.figure_args)
        
        # plot only average or no average
        if use_average:
            df = df.loc[df['Class'] == "Average"]
            if len(df) == 0:
                raise ValueError(f'{metric} has no Average class')
        
        else:            
            df = df.loc[df['Class'] != "Average"]
            df = df.loc[df['Class'] != "Outer"]

        # same marker for each class
        markers = ["o" for i in range(df["Class"].nunique())]

        # if a specific x is provided, use that instead of the default data point
        if x_axis is None:
            x = 'data_point'
        else:
            x = 'x'

        graph = sns.scatterplot(data=df, x=x, y='Value', hue="Class", markers=markers, alpha=0.85, edgecolor=None, palette=self.palette)
        self._set_ylim(graph, metric)
        
        graph.tick_params(left=True, bottom=False)
        graph.set(
            ylabel=self._translate_metric(metric),
        )

        if x_axis is None:
            graph.get_xaxis().set_visible(False)
            graph.set(
                xlim=(0, len(df)/len(markers)),
                xlabel='Predictions',
            )
        else:
            graph.set(
                xlabel=''.join([x_axis[1], ' ', x_axis[0]]),
            )

        return graph


    def ratio(self, data, comparison_target):
        """Plots area or plaque burden ratios, as (Model, GT) pairs"""

        if comparison_target == 'Plaque Burden':
            y = (comparison_target, 'Prediction')
            x = (comparison_target, 'Ground Truth')

        else:
            y = ('Area [mm²]', comparison_target)
            x = ('Area [mm²]', f'{comparison_target} GT')
        
        df = data[[x, y]].copy().astype(float)
        
        if comparison_target == 'Plaque Burden':
            data_max = 0.86
        else:
            data_max = df.max().max()

        # plot needs to be a square shape
        figure_args = self.figure_args.copy()
        avg_size = np.mean(figure_args['figsize'])
        figure_args['figsize'] = (avg_size, avg_size) 

        plt.figure(**self.figure_args)
        graph = sns.scatterplot(data=data, x=x, y=y, marker="o", color="red", s=40, linewidth=0)
        graph = sns.lineplot(x=[0, 1.2 * data_max], y=[0, 1.2 * data_max], ci=None, linewidth=8, color='blue', alpha=0.75)
        graph.axis("scaled")
        graph.set(xlim=(0, 1.2 * data_max), ylim=(0, 1.2 * data_max))
        graph.set(
            xlabel=''.join([x[1], ' ', x[0]]),
            ylabel=''.join([y[1], ' ', y[0]]),
        )
        
        if comparison_target == 'Plaque Burden':
            graph.set(
                xticks=[i/10 for i in range(11)],
                yticks=[i/10 for i in range(11)],
            ) 

        return graph

    def comp_scatter(self, data_dict, metric, comp_class, x_axis=None):
        
        df = pd.DataFrame()

        for tag_i in data_dict:
            df_i = self._format_df(data_dict[tag_i], x=x_axis)
            df_i = df_i.loc[df_i["Metric"] == metric]
            df_i['ID'] = str(tag_i)
            df = pd.concat([df, df_i])

        df.reset_index(inplace=True, drop=True)

        plt.figure(**self.figure_args)
        
        df = df.loc[df['Class'] == comp_class]
        if len(df) == 0:
            raise ValueError(f'{metric} has no {comp_class} class')

        # same marker for each class
        markers = ["o" for i in range(df["Class"].nunique())]

        df['Value'] = df['Value'].astype(float)
        
        # if a specific x is provided, use that instead of the default data point
        if x_axis is None:
            x = 'data_point'
        else:
            x = 'x'
            df['x'] = df['x'].astype(float)

        graph = sns.scatterplot(data=df, x=x, y='Value', hue="ID", markers=markers, alpha=0.45, edgecolor=None, palette=self.palette)
        self._set_ylim(graph, metric)
        
        graph.tick_params(left=True, bottom=False)
        graph.set(
            ylabel=self._translate_metric(metric),
        )

        if x_axis is None:
            graph.get_xaxis().set_visible(False)
            graph.set(
                xlim=(0, df['data_point'].max()),
                xlabel='Predictions',
            )
        else:
            graph.set(
                xlabel=''.join([x_axis[1], ' ', x_axis[0]]),
            )

        return graph


    def comp_violin(self, data_dict, metric):
            
        df = pd.DataFrame()

        for tag_i in data_dict:
            df_i = self._format_df(data_dict[tag_i])
            df_i = df_i.loc[df_i["Metric"] == metric]
            df_i['ID'] = str(tag_i)
            df = pd.concat([df, df_i])

        df.reset_index(inplace=True, drop=True)

        plt.figure(**self.figure_args)
        
        df = df.loc[df['Class'] != "Outer"]
        if len(df) == 0:
            raise ValueError(f'{metric} has no values')

        df['Value'] = df['Value'].astype(float)

        graph = sns.violinplot(data=df, x='Class', y='Value', hue="ID", split=len(data_dict)==2, palette=self.palette)

        self._set_ylim(graph, metric)
        
        graph.tick_params(left=True, bottom=False)
        # graph.get_xaxis().set_visible(False)
        graph.set(
            ylabel=self._translate_metric(metric),
            # xlabel='Predictions',
        )
        graph.tick_params(left=True, bottom=False)
        sns.despine(top=True, left=False, right=True, bottom=True, trim=True)


        return graph


    def comp_box(self, data_dict, metric):
            
        df = pd.DataFrame()

        for tag_i in data_dict:        
            df_i = self._format_df(data_dict[tag_i])
            df_i = df_i.loc[df_i["Metric"] == metric]
            df_i['ID'] = str(tag_i)
            df = pd.concat([df, df_i])

        df.reset_index(inplace=True, drop=True)

        plt.figure(**self.figure_args)
        
        df = df.loc[df['Class'] != "Outer"]
        if len(df) == 0:
            raise ValueError(f'{metric} has no values')

        df['Value'] = df['Value'].astype(float)

        graph = sns.boxplot(data=df, x='Class', y='Value', hue="ID", palette=self.palette)

        self._set_ylim(graph, metric)
        
        graph.tick_params(left=True, bottom=False)
        # graph.get_xaxis().set_visible(False)
        graph.set(
            ylabel=self._translate_metric(metric),
            # xlabel='Predictions',
        )
        graph.tick_params(left=True, bottom=False)
        sns.despine(top=True, left=False, right=True, bottom=True, trim=True)

        return graph    


