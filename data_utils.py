#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat 5 feb 13:50 2022

@author: griffo1
"""

import tensorflow as tf

class DataUtils:
    @staticmethod
    def print_trainable_state(model):
        for i in enumerate(model.model.layers):
            print(i[0], i[1].trainable, i[1].name)