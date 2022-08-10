import sys
import os
import shutil
import itertools

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
    def id_last_training_results_path(id_path, base_name="ivus"):
        """Finds and returns the path of the last training executed in an ID: ivus_0, ivus_1, ivus_2 etc."""

        directory = os.listdir(id_path)

        not_break = True
        n = 0
        last_training = -1
        while not_break:
            find = base_name + "_" + str(n)

            n += 1
                
            if find in directory:
                last_training = find
            else:
                if n > 10:
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

class Comparison:

    #class variables
    row_specs = [
            "Epochs",
            "Net",
            "Depth",
            "Node Structure",
            "Learning Rate",
            "Loss",
            "Optimizer",
            "Base Filters",
            "Kernel Size",
            "Batch Size",
            "Base ID",
            "Multi-Output",
            "Downsizing",
        ]
    
    row_results = [
        "Mean Average IoU",
        "Mean Lumen IoU",
        "Mean Plaque IoU",
        "Mean Vessel IoU",
        "Median Average IoU",
        "Median Lumen IoU",
        "Median Plaque IoU",
        "Median Vessel IoU",
        "IQR Average IoU",
        "IQR Lumen IoU",
        "IQR Plaque IoU",
        "IQR Vessel IoU",
    ]

    complete = [
        "Status",
        "Epochs",
        "Net",
        "Depth",
        "Node Structure",
        "Learning Rate",
        "Loss",
        "Optimizer",
        "Base Filters",
        "Kernel Size",
        "Batch Size",
        "Base ID",
        "Multi-GPU",
        "Multi-Output",
        "Downsizing",
        "Notes",
        "Mean Average IoU",
        "Mean Lumen IoU",
        "Mean Plaque IoU",
        "Mean Vessel IoU",
        "Median Average IoU",
        "Median Lumen IoU",
        "Median Plaque IoU",
        "Median Vessel IoU",
        "IQR Average IoU",
        "IQR Lumen IoU",
        "IQR Plaque IoU",
        "IQR Vessel IoU",
    ]

    @staticmethod
    def _get_specs(specs_path):
        
        specs = pd.read_csv(
            specs_path,
            index_col=0,
            na_values="--",
            dtype=object,
        )
        specs["Base ID"] = specs["Base ID"].fillna("None")
        specs["Multi-GPU"] = specs["Multi-GPU"].fillna(False).replace("X", True)
        specs["Multi-Output"] = specs["Multi-Output"].fillna(False).replace("X", True)
        specs["Mixed Precision"] = specs["Mixed Precision"].fillna(False).replace("X", True)
        specs["Downsizing"] = specs["Downsizing"].fillna("None")
        specs["Multi-channel"] = specs["Multi-channel"].fillna("None") 
        specs["Notes"] = specs["Notes"].fillna("")

        return specs

    @staticmethod
    def _get_results(results_path):
    
        names = pd.MultiIndex.from_product([["Mean", "Median", "IQR"], ["Average IoU", "Lumen IoU", "Plaque IoU", "Vessel IoU"]])
    
        results = pd.read_csv(
            results_path,
            header=None,
            index_col=0,
            names=names,
            skiprows=2,
            dtype=float,
        )
        results.index.name = "ID"
    
        return results
    
    
    @staticmethod
    def _fix_dtypes(specs, kernel_size=int):
        specs = specs.copy()
        to_str = ["Status", "Net", "Loss", "Optimizer", "Notes", "Downsizing", "Base ID", "Multi-channel"]
        to_int = ["Epochs", "Depth", "Node Structure", "Base Filters", "Batch Size"]
        to_bool = ["Multi-GPU", "Multi-Output", "Mixed Precision"]
        to_float = ["Learning Rate", "Median Avg. IoU", "IQR Avg. IoU"]

        if kernel_size == int:
            to_int.append("Kernel Size")
        else:
            to_str.append("Kernel Size")

        specs[to_str] = specs[to_str].astype(str)
        specs[to_float] = specs[to_float].astype(float)
        specs[to_bool] = specs[to_bool].astype(bool)
        specs[to_int] = specs[to_int].astype(int)
    
        return specs
    
    
    @staticmethod
    def _flat_cols(results):
        results = results.copy()
        new_cols = list(map(lambda x: str(x[0] + " " + x[1]), results.columns.to_flat_index()))
        results.columns = new_cols
        return results
    
    
    @staticmethod
    def _make_master(specs, results):

        specs = specs.copy()
        results = results.copy()
    
        specs.drop(columns=["Median Avg. IoU", "IQR Avg. IoU"], inplace=True)
        master_table = pd.concat([specs, results], axis=1)
        master_table.index = master_table.index.astype(int)
    
        return master_table

    @staticmethod
    def get_master(specs_path, results_path):

        specs = Comparison._get_specs(specs_path)
        results = Comparison._get_results(results_path)

        specs = Comparison._fix_dtypes(specs)
        results = Comparison._flat_cols(results)

        master_table = Comparison._make_master(specs, results)

        return master_table

    @staticmethod
    def get_grid(specs_path):

        specs = Comparison._get_specs(specs_path)
        specs = Comparison._fix_dtypes(specs)
        specs = specs.drop(columns=["Status", "Notes", "Median Avg. IoU", "IQR Avg. IoU"])
    
        grid = {k: list(v.unique()) for (k, v) in specs.iteritems()}
    
        for k in grid:
            grid[k].sort()
    
        return grid

    @staticmethod
    def make_grid_df(grid):
        """Returns a DataFrame showing the considered parameters."""

        grid_dict = {k: [str(grid[k])[1:-1].replace("'", "")] for k in grid}

        remove = ["Base ID", "Multi-GPU"]

        for k in remove:
            if k in grid_dict:
                grid_dict.pop(k)

        grid_df = pd.DataFrame.from_dict(grid_dict, orient='index')
        grid_df.columns = ["Considered Values"]
        grid_df.index.name = "Parameter"

        return grid_df


    @staticmethod
    def _get_param_comps(comp_param, master_table, grid, only_completed=True):
        """Returns dictionary of DataFrames containing the comparisons for desired parameters at different states"""

        if only_completed:
            table = master_table.loc[pd.notna(master_table["Mean Average IoU"])].copy()
        else:
            table = master_table.copy()
    
        m_grid = {k: grid[k] for k in grid if k not in comp_param}
        test_consts = [k for k in m_grid]
    
        # fix this
        # vals = [m_grid[k] for k in m_grid]
        # raw_combs = itertools.product(*vals)
        # combs = list(raw_combs)
        # combs_amount = len(combs)
        # print('\r',num*100//combs_amount , 'percent done. Searching for combination', i)
    
        combs = []
        for (i, j) in table[test_consts].iterrows():
            t = tuple(j)
            if t not in combs:
                combs.append(tuple(j))
    
        combs_dict = {}
        for (num, i) in enumerate(combs):
    
            tests = table.loc[(master_table[test_consts] == i).all(1)]
    
            if len(tests) > 1:
                print("found!")
                combs_dict[i] = tests.T.copy()
    
        return combs_dict

    @staticmethod
    def comp_for_specs(specs, grid, master_table, ignore={}):
        """Get comparisons for desired parameters, optionally ignoring some."""

        spec_dict = {}
        comp_summary = {}
    
        n = 1
        specs = list(specs)
        specs.sort()
        for spec in specs:
    
            if isinstance(spec, str):
                spec_set = {spec}
                spec_key = (spec,)
            else:
                spec_set = spec
                spec_key = list(spec)
                spec_key.sort()
                spec_key = tuple(spec_key)
    
            print("Searching for comparisons for", spec)
            combs_dict = Comparison._get_param_comps(spec_set | ignore, master_table, grid, only_completed=True)

            # print("combs_dict:")
            # for i in combs_dict:
            #     print(i, combs_dict[i], sep="\n")
        

            for i in combs_dict:
                comp_summary[n] = [spec, tuple(combs_dict[i].columns.to_list()), i]
                n += 1
                print("Combination", i)
                print("\n", combs_dict[i])
    
            print(len(combs_dict), "combinations found")
    
            spec_dict[spec_key] = combs_dict

        print("comp summary: ")
        for i in comp_summary:
            print(i, comp_summary[i])

        comp_summary = pd.DataFrame.from_dict(comp_summary, orient="index", columns=["Studied Param.", "IDs", "Constant Specifications"])
    
        return spec_dict, comp_summary

    @staticmethod
    def make_tables(id_list, master_table, comp_path):
        
        if not os.path.exists(comp_path):
            os.makedirs(comp_path)

        table = master_table.loc[id_list].T.copy()

        table.to_csv(os.path.join(comp_path, "all.csv"))
        table.loc[Comparison.row_specs].to_latex(os.path.join(comp_path, "specs.tex"))
        table.loc[Comparison.row_results].to_latex(os.path.join(comp_path, "res.tex"))
        table.to_latex(os.path.join(comp_path, "all.tex"))

        return table

    @staticmethod
    def make_auto_comp_tables(spec_dict, suggestions_path):
    
        # suggestions_path = os.path.join(base_target_path, "suggested_comps")
        if not os.path.exists(suggestions_path):
            os.makedirs(suggestions_path)
    
        n = 1
        for k in spec_dict:
            combs_dict = spec_dict[k]
    
            comps_name = list(k)
            comps_name.sort()
            comps_name = tuple(comps_name)
    
            for i in combs_dict:
    
                # if False:
                #     c_list = list(map(lambda x: str(x) + "--", comps_name + i))
                #
                #     name = ""
                #     for c in c_list:
                #         name += c
                #
                #     name = name.lower()
                #     name = name.replace(" ", "")
                #     name = name.replace("0.", "")
                #     name = name[:-2]
                # else:
                #     name = str(n)
    
                name = str(n)
    
                comp_path = os.path.join(suggestions_path, "comp"+name)
                print("saving to", comp_path)

                if not os.path.exists(comp_path):
                    os.makedirs(comp_path)

                combs_dict[i].to_csv(os.path.join(comp_path, "all.csv"))
                combs_dict[i].loc[Comparison.row_specs].to_latex(os.path.join(comp_path, "specs.tex"))
                combs_dict[i].loc[Comparison.row_results].to_latex(os.path.join(comp_path, "res.tex"))
                combs_dict[i].to_latex(os.path.join(comp_path, "all.tex"))
                
                n += 1
    
        print("done!")


class Plotter:

    @staticmethod
    def _retrieve_tables_path(id_list, base_path):
        """Retrieves latest training analysis csv for each training"""

        dic = {}
        for i in id_list:

            # get correct dir
            dir_name = str(i)
            while len(dir_name) < 3:
                dir_name = "0" + dir_name
            dir_name = "id_" + dir_name

            dic[i] = os.path.join(Extractor.id_last_training_results_path(os.path.join(base_path, dir_name)), "analysis.csv")
            
        return dic

    @staticmethod
    def _load_analysis_table(analysis_paths):
        dic = {id: pd.read_csv(analysis_paths[id], index_col=[0]) for id in analysis_paths}
        return dic

    @staticmethod
    def _make_comp_df(analysis_tables):
        """Constructs a comparison DataFrame in the appropriate format for plotting comparison plots"""

        analysis_tables = {str(id): analysis_tables[id].copy() for id in analysis_tables}

        for id in analysis_tables:
            analysis_tables[id]["ID"] = [id for n in range(len(analysis_tables[id]))]

        df = pd.concat([analysis_tables[id] for id in analysis_tables])

        df = df[["Average", "Lumen", "Plaque", "Vessel", "ID"]]

        # graph = sns.violinplot(data=df, )

        df2 = pd.DataFrame()
        for j in ["Average", "Lumen", "Plaque", "Vessel"]:
            df_aux = df[[j, "ID"]].copy()
            df_aux.columns = ["IoU", "ID"]
            df_aux["Class"] = [j for n in range(len(df_aux))]

            df2 = pd.concat([df2, df_aux])

        df2 = df2.copy()

        return df2

    @staticmethod
    def violin(id_list, base_path):

        analysis_paths = Plotter._retrieve_tables_path(id_list, base_path)
        analysis_tables = Plotter._load_analysis_table(analysis_paths)
        df = Plotter._make_comp_df(analysis_tables)

        plt.figure()
        
        if len(id_list) == 2:
            graph = sns.violinplot(data=df, x="Class", y="IoU", hue="ID", split=True)
        else:
            graph = sns.violinplot(data=df, x="Class", y="IoU", hue="ID")

        return graph

    @staticmethod
    def all_comps(comp_summary, base_path, suggestions_path):
        
        for (i, j) in comp_summary.iterrows():

            comp_path = os.path.join(suggestions_path, "comp"+str(i))
            if not os.path.exists(comp_path):
                os.makedirs(comp_path)

            id_list = j["IDs"]

            print(f"drawing plot for {id_list}")
            graph = Plotter.violin(id_list, base_path)
            graph.get_figure().savefig(os.path.join(comp_path, "violin.png"), format="png", dpi=400)
            plt.close()
            print(f"saving to {comp_path}")

        print("done!")

        return

    @staticmethod
    def master_table_plot(master_table):

        table = master_table[["Median Average IoU", "IQR Average IoU"]].loc[pd.notna(master_table["Mean Average IoU"])]
        table.sort_values(by="Median Average IoU", inplace=True)
        # table["ID"] = [str(i) for i in table.index]
        table["ID"] = [i for i in table.index]


        df = table
        df2 = pd.DataFrame()
        for j in ["Median Average IoU", "IQR Average IoU"]:
            df_aux = df[[j, "ID"]].copy()
            df_aux.columns = ["IoU", "ID"]
            df_aux["Metric"] = [j for n in range(len(df_aux))]

            df2 = pd.concat([df2, df_aux])

        df2.index = [i for i in range(len(df2))]
        df2 = df2.copy()

        print(df2)
        plt.figure()
        graph = sns.barplot(data=df2, x="ID", y="IoU", hue="Metric", order=table.index)
        graph.set(ylim=(0, 1.03))
        # graph.set_aspect(9/16)

        graph.tick_params(axis="x", which="major", labelsize=7, rotation=45)
        graph.tick_params(axis="y", which="major", labelsize=8, rotation=0)



        return