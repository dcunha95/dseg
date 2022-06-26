import sys
import os
import shutil
import itertools

import pandas as pd
import seaborn as sns
import numpy as np


class Extractor:
    @staticmethod
    def _process_log(log):
        """Retrieves training info from log file and returns a DataFrame."""

        with open(log, "r") as f:
            data = f.read()

        data = data.replace("\x08", "")
        data = [i for i in data.split("\n") if "loss:" in i]

        if data == []:
            return pd.DataFrame()

        pdata = []
        for i in data:
            j = i.split(" ")[0].split("/")
            if j[0] == j[-1] and len(j) == 2 and "ETA" not in i:
                # print(j)
                # print(i)
                pdata.append(i)

        names = []
        j = pdata[0].split(":")
        jl = len(j)
        for k in range(jl - 1):
            names.append(j[k].split(" ")[-1])

        # print(names)

        ddata = {}
        for n in names:
            ddata[n] = []

        # print(ddata)
        for i in pdata:
            j = i.split(" ")
            for k in enumerate(j):
                if k[1][:-1] in names:
                    ddata[k[1][:-1]].append(j[k[0] + 1])

        # print(pdata)
        # print(ddata)
        df = pd.DataFrame(ddata)
        df = df.astype(float)
        df["epoch"] = [*range(1, len(df) + 1)]
        df = df.set_index("epoch")
        return df

    @staticmethod
    def get_training_info(path):
        """Given a path, returns a DataFrame containing the training info from all logs."""

        logs = [log for log in os.listdir(path) if ".out" in log]
        logs.sort()

        datum = [Extractor._process_log(os.path.join(path, log)) for log in logs]

        epochs = 0
        for i in range(len(datum)):
            datum[i].index = [epochs + 1 + j for j in range(len(datum[i]))]
            epochs += len(datum[i])

        if len(datum) != 1:
            data = pd.concat(datum)

        else:
            data = datum[0]

        return data

    @staticmethod
    def plot_training(data, path):

        training = pd.DataFrame()

        training["Loss"] = data["loss"]
        training["Val. Loss"] = data["val_loss"]

        if "lumen_loss" in data.columns or "vessel_loss" in data.columns:
            multi = True
        else:
            multi = False

        if multi:

            training["Lumen Loss"] = data["lumen_loss"]
            training["Lumen Val. Loss"] = data["val_lumen_loss"]

            training["Vessel Loss"] = data["vessel_loss"]
            training["Vessel Val. Loss"] = data["val_vessel_loss"]

            training["Lumen Mean IoU"] = data["lumen_mean_io_u"]
            training["Lumen Val. Mean IoU"] = data["val_lumen_mean_io_u"]

            training["Vessel Mean IoU"] = data["vessel_mean_io_u_1"]
            training["Vessel Val. Mean IoU"] = data["val_vessel_mean_io_u_1"]

            plots = [
                (
                    "lumen_training_results.png",
                    [
                        "Lumen Loss",
                        "Lumen Val. Loss",
                        "Lumen Mean IoU",
                        "Lumen Val. Mean IoU",
                    ],
                ),
                (
                    "vessel_training_results.png",
                    [
                        "Vessel Loss",
                        "Vessel Val. Loss",
                        "Vessel Mean IoU",
                        "Vessel Val. Mean IoU",
                    ],
                ),
                (
                    "training_results.png",
                    [
                        "Loss",
                        "Val. Loss",
                        "Lumen Val. Mean IoU",
                        "Vessel Val. Mean IoU",
                    ],
                ),
            ]

        else:

            training["Mean IoU"] = data["mean_io_u"]
            training["Val. Mean IoU"] = data["val_mean_io_u"]

            plots = [
                (
                    "training_results.png",
                    ["Loss", "Val. Loss", "Mean IoU", "Val. Mean IoU"],
                ),
            ]

        training["Epoch"] = np.arange(1, len(training) + 1)
        training.set_index("Epoch", inplace=True)

        sns.set_theme(style="ticks")

        for plot in plots:
            graph = sns.lineplot(
                data=training[plot[1]],
                palette="bright",
                markers=True,
                dashes=False,
            )
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
            graph.get_figure().savefig(os.path.join(path, plot[0]), format="png", dpi=800)
            graph.get_figure().clf()

        return training

    @staticmethod
    def id_last_training_results_path(id_path):
        """Finds and returns the path of the last training executed in an ID: ivus_0, ivus_1, ivus_2 etc."""

        dir = os.listdir(id_path)

        not_break = True
        n = 0
        last_training = -1
        while not_break:
            find = "ivus_" + str(n)

            if find in dir:
                n += 1
                last_training = find
            else:
                not_break = False

        if last_training == -1:
            raise FileNotFoundError(f"No trainings found in {id_path}")

        return os.path.join(id_path, last_training)

    @staticmethod
    def get_results_analysis(path):
        """Reads analysis_summary.csv and analysis.csv files and returns formatted DataFrames."""

        data_info = pd.read_csv(os.path.join(path, "analysis_summary.csv"))
        data = pd.read_csv(os.path.join(path, "analysis.csv"))

        # data_info processing
        data_info.set_index("Unnamed: 0", drop=True, inplace=True)
        data_info.index.name = ""

        data_info.loc["IQR"] = data_info.loc["75%"] - data_info.loc["25%"]

        data_info.index = [i.lower() for i in data_info.index]
        data_info = data_info.loc[["count", "mean", "std", "iqr", "min", "25%", "50%", "75%", "max"]]
        data_info = data_info.loc[:, :].astype("object")
        data_info.loc["count"] = data_info.loc["count"].map("{:.0f}".format)
        data_info.index = ["Count", "Mean", "Std", "IQR", "Min", "25%", "50%", "75%", "Max"]

        for i in data_info.iteritems():
            data.loc[:, i[0]] = data[i[0]].map("{:.4f}".format)
            data_info.loc["Mean":"Max", i[0]] = data_info.loc["Mean":"Max", i[0]].map("{:.4f}".format)

        cols = data_info.columns.to_list()

        for (i, j) in enumerate(data_info.columns):
            if j == "Plaque Burden Model/GT Ratio":
                cols[i] = "PB. Ratio"

        data_info.columns = cols

        # data processing
        data.drop(columns="Unnamed: 0", inplace=True)

        cols = data.columns.to_list()

        for (i, j) in enumerate(data.columns):
            if j == "Plaque Burden Model/GT Ratio":
                cols[i] = "PB. Ratio"

        data.columns = cols

        # data.columns = cols

        data = data.astype("str")
        data_info = data_info.astype("str")

        print(data.head(5))
        print(data_info)

        analysis = {}
        analysis["data_info"] = data_info
        analysis["data"] = data

        return analysis

    @staticmethod
    def make_tables(path, analysis):
        """Saves latex tables at the specified folder."""

        save_folder = os.path.join(path, "latex")

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        raw_columns = ["Average", "Lumen", "Plaque", "Vessel", "Lumen Area Ratio", "Plaque Area Ratio", "Vessel Area Ratio", "PB. Ratio"]
        raw_mod_columns = ["Average", "Lumen", "Plaque", "Vessel", "L. Area Ratio", "P. Area Ratio", "V. Area Ratio", "PB. Ratio"]

        res_columns = ["Average", "Lumen", "Plaque", "Vessel", "Lumen Area Ratio", "Plaque Area Ratio", "Vessel Area Ratio", "PB. Ratio"]
        res_mod_columns = ["Average", "Lumen", "Plaque", "Vessel", "L. Area Ratio", "P. Area Ratio", "V. Area Ratio", "PB. Ratio"]

        raw = analysis["data"][raw_columns]
        raw.columns = [raw_mod_columns]
        raw.to_latex(save_folder + "/results_raw.tex", longtable=True)

        res = analysis["data_info"][res_columns]
        res.columns = [res_mod_columns]
        res.to_latex(save_folder + "/results.tex")

        res[["Average", "Lumen", "Plaque", "Vessel"]].to_latex(save_folder + "/results_primary.tex")
        res[["L. Area Ratio", "P. Area Ratio", "V. Area Ratio", "PB. Ratio"]].to_latex(save_folder + "/results_secondary.tex")

    @staticmethod
    def make_results_table(all_analysis):
        """Draws results table with all ids"""

        index = []
        cols = {i: [] for i in itertools.product(["Mean", "Median", "IQR"], ["Average IoU", "Lumen IoU", "Plaque IoU", "Vessel IoU"])}

        # making the index
        for path in all_analysis:
            done = False
            p = path
            while not done:
                p, n = os.path.split(p)
                if "id_" in n:
                    done = True

            n = int(n.replace("id_", ""))

            index.append(n)

            for (i, j) in cols:
                cols[(i, j)].append(all_analysis[path]["data_info"].loc[i.replace("Median", "50%"), j.replace(" IoU", "")])

        df = pd.DataFrame.from_dict(cols, orient="columns")
        df.index = index
        df.index.name = "ID"

        n_max = max(index)
        for i in range(1, n_max + 1):
            if i not in index:
                df.loc[i] = [0 for j in range(len(df.columns))]

        df.sort_index(inplace=True)

        return df

    @staticmethod
    def extract(input_path, target_path):
        """Copy relevant png and tex files to target_path"""

        latex_folder = os.path.join(input_path, "latex")
        plots_folder = os.path.join(input_path, "plots")

        # latex tables
        for file in os.listdir(latex_folder):
            source = os.path.join(latex_folder, file)
            destination = os.path.join(target_path, file)
            shutil.copy(source, destination)

        # plots
        relevant_plots = [
            "iou_scatter_avg.png",
            "iou_scatter.png",
            "iou_violin_bw_0.2_avg.png",
        ]

        for plot in relevant_plots:
            source = os.path.join(plots_folder, plot)
            destination = os.path.join(target_path, plot)
            shutil.copy(source, destination)

        # training plot
        source = os.path.join(os.path.split(input_path)[0], "training_results.png")
        destination = os.path.join(target_path, "training_results.png")
        shutil.copy(source, destination)
