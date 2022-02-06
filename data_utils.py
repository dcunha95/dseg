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


class DataUtils:
    @staticmethod
    def print_trainable_state(model):
        for i in enumerate(model.model.layers):
            print(i[0], i[1].trainable, i[1].name)

    @staticmethod
    def save_model(model):

        path = model.model_name + "/" + model.model_name + ".dseg"

        with h5py.File(path, "w") as file:
            for i in model.__dict__:
                if isinstance(model.__dict__[i], pd.DataFrame):
                    pass
                elif i == "model":
                    tf.keras.models.save_model(model=model.model, filepath=file)
                elif isinstance(model.__dict__[i], str):
                    file[i] = model.__dict__[i]

        for i in model.__dict__:
            if isinstance(model.__dict__[i], pd.DataFrame):
                model.__dict__[i].to_hdf(path, key=i)




    @staticmethod
    def load_model(path):
        #fix_this_mess
        if path.split('.')[-1] == "pkl":
            with open(path, "rb") as file:
                model = pickle.load(file)
        else:
            with h5py.File(path, "r") as file:
                model = file['model']

        return model
