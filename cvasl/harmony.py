
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and
the Amsterdam University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

Most seperated values processing
is in the seperated modeul, hoever, this file contains functions
for processing csv and tsv
files as they relate to specific harmonization algorithms.
Essentially this module has been made
so it can be called in environments
compatible with common
harmonization algorithms which often require
older versions of python and numpy than usual in 2023.
"""

import os
import copy
import glob
import pandas as pd
import numpy as np


def log_out_columns(dataframe, column_list):
    for column in column_list:
        dataframe[column] = np.log(dataframe[column])
    return dataframe


def split_frame_half_balanced_by_column(frame, column):
    """
    This is function is made for a dataframe you want to split
    on a columns with continous values e.g. age.
    """
    df = frame.sort_values(column).reset_index()
    rng_even = range(0, len(df), 2)
    rng_odd = range(1, len(df), 2)
    even_rows = df.iloc[rng_even]
    odd_rows = df.iloc[rng_odd]
    return even_rows, odd_rows


def top_and_bottom_by_column(frame, column):
    """This is useful in cases where you want to split on a columns
    with continous values e.g. age.; and upi
    want the highest and lowest values seperated"""
    df = frame.sort_values(column)
    len_first_half = len(df) // 2
    top = df.iloc[:len_first_half]
    bottom = df.iloc[len_first_half:]
    return top, bottom


def prep_for_neurocombat(dataframename1, dataframename2):
    """
    This function takes two dataframes in the cvasl format,
    then turns them into three items needed for the
    neurocombat algorithm with re-identification.
    """
    two_selection = dataframename2.drop(['Unnamed: 0'], axis=1)
    one_selection = dataframename1.drop(['Unnamed: 0'], axis=1)
    one_selection = one_selection.set_index('participant_id')
    two_selection = two_selection.set_index('participant_id')
    one_selection = one_selection.T
    two_selection = two_selection.T
    both_togetherF = pd.concat(
        [one_selection, two_selection],
        axis=1,
        join="inner",
    )
    print("Nan count", both_togetherF.isna().sum().sum())
    features_only = both_togetherF[2:]
    dictionary_features_len = len(features_only.T.columns)
    number = 0
    made_keys = []
    made_vals = []
    for n in features_only.T.columns:

        made_keys.append(number)
        made_vals.append(n)
        number += 1
    feature_dictF = dict(map(lambda i, j: (i, j), made_keys, made_vals))
    ftF = features_only.reset_index()
    ftF = ftF.rename(columns={"index": "A"})
    ftF = ftF.drop(['A'], axis=1)
    ftF = ftF.dropna()
    btF = both_togetherF.reset_index()
    btF = btF.rename(columns={"index": "A"})
    btF = btF.drop(['A'], axis=1)
    btF = btF.dropna()
    len1 = len(one_selection.columns)
    len2 = len(two_selection.columns)
    return both_togetherF, ftF, btF, feature_dictF, len1, len2


def make_topper(btF, row0, row1):
    """
    This function makes top rows for something harmonized
    out of the btF part produced with
    prep_for_neurocombat(dataframename1, dataframename2)
    """
    topperF = btF.head(2)
    topperF = topperF.rename_axis(None, axis="columns")
    topperF = topperF.reset_index(drop=False)
    topperF = topperF.rename(columns={"index": "char"})
    topperF['char'][0] = row0  # 'age'
    topperF['char'][1] = row1  # 'sex'
    return topperF
