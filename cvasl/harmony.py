
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and
the Amsterdam University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for processing csv and tsv
files as they relate to specific common harmonization algorithms.
Most seperated values processing is in the seperated module,
however, this  this module has been made so it can be called in environments
compatible with common harmonization algorithms which often require
older versions of python, pandas and numpy than usual in 2023.
"""

import os
import copy
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def log_out_columns(dataframe, column_list):
    """
    This function recodes changes specified
    column values in a dataframe to a log
    of the values, which can make overall
    distributions change.

    :param dataframe: dataframe variable
    :type dataframe: str
    :param column_list: column names
    :type column_list: list

    :returns: dataframe with different (log) values in specified columns
    :rtype: pandas.dataFrame
    """
    frame = dataframe.copy()
    for column in column_list:
        frame[column] = np.log(dataframe[column])
    return frame


def split_frame_half_balanced_by_column(frame, column):
    """
    This is function is made for a dataframe you want to split
    on a columns with continous values e.g. age.; and returns
    two dataframes in which the values in this column are
    about equally distributed e.g. average age over both frames,
    if age is column variable, will be similar

    :param dataframe: frame variable
    :type frame: str
    :param column: column name
    :type column: Series

    :returns: dataframes evenly idstributed on values in specified column
    :rtype: pandas.dataFrame
    """
    df = frame.sort_values(column).reset_index()
    rng_even = range(0, len(df), 2)
    rng_odd = range(1, len(df), 2)
    even_rows = df.iloc[rng_even]
    odd_rows = df.iloc[rng_odd]
    return even_rows, odd_rows


def top_and_bottom_by_column(frame, column):
    """
    This is useful in cases where you want to split on a columns
    with continous values e.g. age.; and upi
    want the highest and lowest values seperated

    :param dataframe: frame variable
    :type frame: str
    :param column: column name
    :type column: Series

    :returns: dataframes unevenly distributed on values in specified column
    :rtype: `~pandas.DataFrame`
    """
    df = frame.sort_values(column)
    len_first_half = len(df) // 2
    top = df.iloc[:len_first_half]
    bottom = df.iloc[len_first_half:]
    return top, bottom


def prep_for_neurocombat(dataframe1, dataframe2):
    """
    This function takes two dataframes in the cvasl format,
    then turns them into the items needed for the
    neurocombat algorithm with re-identification.

    :param dataframe1: frame variable
    :type frame: `~pandas.DataFrame`
    :param dataframe2: frame variable
    :type frame: `~pandas.DataFrame`

    :returns: dataframes for neurocombat algorithm and ints of some legnths
    :rtype: tuple
    """
    # TODO:(makeda) make so it can take frame name or frame
    two_selection = dataframe2.drop(['Unnamed: 0'], axis=1)
    one_selection = dataframe1.drop(['Unnamed: 0'], axis=1)
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
    out of the btF part produced by the prep_for_neurocombat function
    i.e. prep_for_neurocombat(dataframename1, dataframename2)

    :param btF: frame variable produced in prep_for_neurocombat
    :type btF: `~pandas.DataFrame`
    :param row0: frame column removed i.e. age or sex
    :type row0: str
    :param row1: frame column removed i.e. age or sex
    :type row1: str

    :returns: dataframe called TopperF to add back
    :rtype: `~pandas.DataFrame`
    """
    topperF = btF.head(2)
    topperF = topperF.rename_axis(None, axis="columns")
    topperF = topperF.reset_index(drop=False)
    topperF = topperF.rename(columns={"index": "char"})
    topperF['char'][0] = row0  # 'age'
    topperF['char'][1] = row1  # 'sex'
    return topperF


def compare_harm_multi_site_violins(
        unharmonized_df,
        harmonized_df,
        feature_list
):
    """
    Create a violin plot on multisite harmonization by features.
    """
    for feature in feature_list:
        complete_merge = pd.concat(
            [unharmonized_df, harmonized_df]).reset_index(drop=True)
        complete_merge[feature] = complete_merge[feature].astype('float64')
        sns.set_style("whitegrid")
        y_axis = feature
        g = sns.FacetGrid(complete_merge, col="batch")
        g.map(
            sns.violinplot,
            'harmonization',
            y_axis,
            order=["UH", "H"],
            palette=["b", "g", "pink",],
            alpha=0.3
        )
        lowest_on_graph = complete_merge[y_axis].min() - 0.5
        plt.ylim(
            lowest_on_graph,
            complete_merge[y_axis].max() * 1.5)
        plt.show()


def compare_harm_one_site_violins(
        unharmonized_df,
        harmonized_df,
        feature_list,
        chosen_feature="sex"
):
    """
    Create a violin plot on single site harmonization by features,
    split on a binary feature of choice which defaults to sex.
    """
    for feat in feature_list:
        complete_merg = pd.concat(
            [unharmonized_df, harmonized_df]).reset_index(drop=True)
        complete_merg[feat] = complete_merg[feat].astype('float64')
        sns.set_style("whitegrid")
        y_axis = feat
        g = sns.catplot(
            data=complete_merg,
            x='harmonization', y=y_axis, hue=chosen_feature,
            split=True, inner='quartile', kind='violin',
            height=5, aspect=0.6, palette=['pink', 'blue'], alpha=0.4)

        lowest_on_graph = complete_merg[y_axis].min() - 0.5
        plt.ylim((lowest_on_graph, complete_merg[y_axis].max() * 1.5))
        plt.title(feat)
        plt.show()
