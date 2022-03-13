#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 16:13:39 2022

@author: griffo1
"""

import pandas as pd
import dseg.data_collection as dc


class DataManipulator:
    def __init__(self, setup):
        self.setup = setup

    @staticmethod
    def df_div(
        df,
        factor,
    ):
        frac = df.iloc[::factor, :]
        frac.index = [i for i in range(len(frac))]

        return frac 

    def load_dataset(
        self,
        randomize=True,
        file_name="dataset/reduced.csv",
    ):

        # prepare paths
        files = dc.load_files(file_name, randomize=randomize, seed=1337)

        # separate dataset in train, val and test
        (
            trn_dataset,
            val_dataset,
            tst_dataset,
        ) = dc.split_files(files, self.setup.pipeline_config.dataset_percent, split=self.setup.pipeline_config.split)

        stat_dataset = val_dataset

        return trn_dataset, val_dataset, tst_dataset, stat_dataset

    def load_partitions(
        self,
        train_file_name="dataset/dataset_1MC_train.csv",
        val_file_name="dataset/dataset_1MC_val.csv",
        test_file_name="dataset/dataset_1MC_test.csv",
        stat_file_name="dataset/dataset_1MC_stat.csv",
    ):

        # prepare paths
        train_files = dc.load_files(train_file_name, randomize=False)
        val_files = dc.load_files(val_file_name, randomize=False)
        test_files = dc.load_files(test_file_name, randomize=False)
        stat_files = dc.load_files(stat_file_name, randomize=False)

        trn_dataset = dc.split_files(train_files, self.setup.pipeline_config.dataset_percent, do_split=False)
        val_dataset = dc.split_files(val_files, self.setup.pipeline_config.dataset_percent, do_split=False)
        tst_dataset = dc.split_files(test_files, self.setup.pipeline_config.dataset_percent, do_split=False)
        stat_dataset = dc.split_files(stat_files, 1, do_split=False)

        return trn_dataset, val_dataset, tst_dataset, stat_dataset
