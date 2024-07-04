
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
from itertools import permutations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.mixture import GaussianMixture
# from scipy.stats import ranksums, ttest_ind
# from scipy.stats import ttest_rel, ks_2samp, anderson_ksamp
# import neuroCombat as nC


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
    if 'Unnamed: 0' in dataframe2.columns:
        two_selection = dataframe2.drop(['Unnamed: 0'], axis=1)
    else:
        two_selection = dataframe2
    if 'Unnamed: 0' in dataframe1.columns:
        one_selection = dataframe1.drop(['Unnamed: 0'], axis=1)
    else:
        one_selection = dataframe1
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


def prep_for_neurocombat_5way(
        dataframe1,
        dataframe2,
        dataframe3,
        dataframe4,
        dataframe5):
    """
    This function takes five dataframes in the cvasl format,
    then turns them into the items needed for the
    neurocombat algorithm with re-identification.

    :param dataframe1: frame variable
    :type frame: `~pandas.DataFrame`
    :param dataframe2: frame variable
    :type frame: `~pandas.DataFrame`
    :param dataframe3: frame variable
    :type frame: `~pandas.DataFrame`
    :param dataframe4: frame variable
    :type frame: `~pandas.DataFrame`
    :param dataframe5: frame variable
    :type frame: `~pandas.DataFrame`


    :returns: dataframes for neurocombat algorithm and ints of some legnths
    :rtype: tuple
    """
    # TODO:(makeda) make so it can take frame name or frame

    two_selection = dataframe2
    one_selection = dataframe1
    three_selection = dataframe3
    four_selection = dataframe4
    five_selection = dataframe5
    # set index to participant IDs
    one_selection = one_selection.set_index('participant_id')
    two_selection = two_selection.set_index('participant_id')
    three_selection = three_selection.set_index('participant_id')
    four_selection = four_selection.set_index('participant_id')
    five_selection = five_selection.set_index('participant_id')

    # turn dataframes on side
    one_selection = one_selection.T
    two_selection = two_selection.T
    three_selection = three_selection.T
    four_selection = four_selection.T
    five_selection = five_selection.T

    # concat the two dataframes
    all_togetherF = pd.concat(
        [one_selection,
         two_selection,
         three_selection,
         four_selection,
         five_selection],
        axis=1,
        join="inner",
    )

    # create a feautures only frame (no age, no sex)
    features_only = all_togetherF[2:]
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
    btF = all_togetherF.reset_index()
    btF = btF.rename(columns={"index": "A"})
    btF = btF.drop(['A'], axis=1)
    btF = btF.dropna()
    len1 = len(one_selection.columns)
    len2 = len(two_selection.columns)
    len3 = len(three_selection.columns)
    len4 = len(four_selection.columns)
    len5 = len(five_selection.columns)

    return all_togetherF, ftF, btF, feature_dictF, len1, len2, len3, len4, len5


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
        feature_list,
        batch_column='site'
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
        g = sns.FacetGrid(complete_merge, col=batch_column)
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


def negative_harm_outcomes(
    folder,
    file_extension,
    number_columns=[
        'sex',
        'gm_vol',
        'wm_vol',
        'csf_vol',
        'gm_icvratio',
        'gmwm_icvratio',
        'wmhvol_wmvol',
        'wmh_count',
        'deepwm_b_cov',
        'aca_b_cov',
        'mca_b_cov',
        'pca_b_cov',
        'totalgm_b_cov',
        'deepwm_b_cbf',
        'aca_b_cbf',
        'mca_b_cbf',
        'pca_b_cbf',
        'totalgm_b_cbf',]
):
    """
    This function given a directory will
    search all subdirectory for noted file extension
    If all files are harmonization outcome files
    it will then return a list of files with negative values,
    and print off information about negatives in all files.
    """
    files = '**/*.' + file_extension

    suspects = glob.glob(
        os.path.join(folder, files),
        recursive=True,
    )
    list_negs = []
    for file in suspects:
        read = pd.read_csv(file)
        read.columns = read.columns.str.lower()
        print(file)
        print((read[number_columns] < 0).sum())
        if ((read[number_columns] < 0).sum().sum()) > 0:
            list_negs.append(file)
    return list_negs


def show_diff_on_var(
        dataset1,
        name_dataset1,
        dataset2,
        name_dataset2,
        var1,
        var2):
    dataset1['set'] = name_dataset1
    dataset2['set'] = name_dataset2
    mixer = pd.concat([dataset1, dataset2])
    jp1 = sns.jointplot(
        x=mixer[var1],
        y=mixer[var2],
        hue=mixer['set'],
        alpha=0.3,
        space=0,
        ratio=4)


def show_diff_on_var3(
        dataset1,
        name_dataset1,
        dataset2,
        name_dataset2,
        dataset3,
        name_dataset3,
        var1,
        var2,
        ):
    dataset1['set'] = name_dataset1
    dataset2['set'] = name_dataset2
    dataset3['set'] = name_dataset3
    mixer = pd.concat([dataset1, dataset2, dataset3])
    jp1 = sns.jointplot(
        x=mixer[var1],
        y=mixer[var2],
        hue=mixer['set'],
        alpha=0.2,
        space=0,
        ratio=4,
        )


def show_diff_on_var5(
        dataset1,
        name_dataset1,
        dataset2,
        name_dataset2,
        dataset3,
        name_dataset3,
        dataset4,
        name_dataset4,
        dataset5,
        name_dataset5,
        var1,
        var2):
    dataset1['set'] = name_dataset1
    dataset2['set'] = name_dataset2
    dataset3['set'] = name_dataset3
    dataset4['set'] = name_dataset4
    dataset5['set'] = name_dataset5
    mixer = pd.concat([dataset1, dataset2, dataset3, dataset4, dataset5])
    jp1 = sns.jointplot(
        x=mixer[var1],
        y=mixer[var2],
        hue=mixer['set'],
        alpha=0.2,
        space=0,
        ratio=4)


def increment_keys(input_dict, chosen_value=1):
    """
    This function increments all keys in dictionary by a certain chosen value.
    """
    return {key + chosen_value: value for key, value in input_dict.items()}
