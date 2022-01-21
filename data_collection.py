# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:34:20 2021

@author: danie
"""

import os
import random
import json
import csv
import pandas as pd


#%% collect from directory

# one at a time
def get_path(input_dir):

    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )

    files = sorted(os.listdir(input_dir))

    return input_img_paths, files


# compares files found in multiple dirs and return them in couples (if a couple is found)
def organize_path(input_dirs, target_dirs):

    input_paths = []
    for i in input_dirs:
        path = get_path(i)
        for j in range(len(path[0])):
            input_paths += [[path[1][j], path[0][j]]]
    target_paths = []
    for i in target_dirs:
        path = get_path(i)
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


def save_csv(file_name, files):

    with open(file_name, "w", newline="") as fp:
        writer = csv.writer(fp)

        x = [None for i in range(len(files) + 1)]

        x[0] = ["Imagem", "Média", "Externo", "Lúmen", "Parede"]

        for i in range(len(files)):
            x[i + 1] = [i] + files[i]
        writer.writerows(x)
    return


def save_files(file_name, files):

    # with open(file_name, "w") as fp:
    #     json.dump(files, fp, indent=2)

    files.to_csv(file_name)

    return


def load_files(file_name, randomize=True, seed=1337):

    # with open(file_name, "r") as fp:
    #     files = json.load(fp)
    # if randomize:
    #     random.Random(seed).shuffle(files)

    files = pd.read_csv(file_name)
    files.drop(columns=["Unnamed: 0"], inplace=True)

    if randomize:
        files = files.sample(frac=1, random_state=seed).reset_index(drop=True)
    return files


#%% dataset splitting

# do a train/validation split for files
def split_files(files, dataset_percent, do_split=True, split=[0.6, 0.2, 0.2]):

    if do_split:

        train_split = split[0]
        validation_split = split[1]
        test_split = split[2]

        trn_amount = int(train_split * int(dataset_percent * len(files)))
        val_amount = int(validation_split * int(dataset_percent * len(files)))
        test_amount = int(test_split * int(dataset_percent * len(files)))

        trn_dataset = files.iloc[-(val_amount + test_amount + trn_amount): -(val_amount + test_amount)]
        val_dataset = files.iloc[-(val_amount + test_amount) : -test_amount]
        tst_dataset = files.iloc[-test_amount:]

        return (
            trn_dataset,
            val_dataset,
            tst_dataset,
        )
    else:
        
        amount = int(dataset_percent * len(files))
        
        dataset = files.iloc[:amount]
        
        return dataset


#%%

# toma o maior batch_size compatível com o tamanho do dataset de predicao
def get_pred_batch_size(paths, max_batch_size):

    size = 1
    length = len(paths)

    for i in range(1, max_batch_size + 1):
        if length % i == 0:
            size = i
    return size
