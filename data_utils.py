#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat 5 feb 13:50 2022

@author: griffo1
"""
import pandas as pd
import tensorflow as tf
import pickle
import h5py
import dseg.model
import dseg.setup


class DataUtils:
    @staticmethod
    def print_trainable_state(
            model
    ):
        for i in enumerate(model.model.layers):
            print(i[0], i[1].trainable, i[1].name)


    @staticmethod
    def save_model(
            model
    ):

        path = model.model_name + "/" + model.model_name + ".dseg"

        # fix_this_mess
        with h5py.File(path, "w") as file:
            for i in model.__dict__:
                if isinstance(model.__dict__[i], pd.DataFrame):
                    pass
                elif i == "model":
                    tf.keras.models.save_model(model=model.model, filepath=file)

                elif isinstance(model.__dict__[i], str):
                    file[i] = model.__dict__[i]

            setup_dic = model.setup.dict
            for i in setup_dic:
                s = 'setup/' + i + '/'
                for j in setup_dic[i]:
                    print(i, j, setup_dic[i][j], sep='\t\t\t')
                    file[s + j] = setup_dic[i][j]

        for i in model.__dict__:
            if isinstance(model.__dict__[i], pd.DataFrame):
                model.__dict__[i].to_hdf(path, key=i)

    @staticmethod
    def load_model(
            path
    ):
        # fix_this_mess

        # setup = dseg.setup.Setup(pipeline_config=dseg.)

        model = dseg.model.Segmenter()

        with h5py.File(path, "r") as file:
            pass
