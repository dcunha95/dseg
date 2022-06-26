import sys
import os

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
                ("lumen_training_results.png", ["Lumen Loss", "Lumen Val. Loss", "Lumen Mean IoU", "Lumen Val. Mean IoU"]),
                ("vessel_training_results.png", ["Vessel Loss", "Vessel Val. Loss", "Vessel Mean IoU", "Vessel Val. Mean IoU"]),
                ("training_results.png", ["Loss", "Val. Loss", "Lumen Val. Mean IoU", "Vessel Val. Mean IoU"]),
            ]

        else:

            training["Mean IoU"] = data["mean_io_u"]
            training["Val. Mean IoU"] = data["val_mean_io_u"]

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
            graph.get_figure().savefig(os.path.join(path, plot[0]), format="png", dpi=800)
            graph.get_figure().clf()

        return training
