
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and Stichting VUMC.
Licensed under <TBA>. See LICENSE for details.

This file contains functions for processing csv and tsv
files towards correct formats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import os


def recode_sex(whole_dataframe, string_for_sex):
    """
    This function recodes sex into a new column if there
    are two possible values. It maintains numerical order
    but changes the values to 0 and 1. The new column is
    called 'sex_encoded'. Note sex should be encoded
    in numbers i.e. ints or floats

    :param whole_dataframe: dataframe variable
    :type whole_dataframe: str
    :param string_for_sex: column name written in singe qoutes
    :type string_for_sex: str

    :returns: dataframe with sex encoded colum
    :rtype: pandas.dataFrame

    """
    new_dataframe = whole_dataframe.copy()
    dataframe_column = whole_dataframe[string_for_sex]
    recoded = dataframe_column.copy()
    if 999 in set(recoded.unique()):
        recoded = recoded.replace(999, 'NaN')
    if '999' in set(recoded.unique()):
        recoded = recoded.replace('999', 'NaN')
    if 'M' in set(recoded.unique()):
        recoded = recoded.replace('M', 0)
    if 'F' in set(recoded.unique()):
        recoded = recoded.replace('F', 1)
    if len(dataframe_column.unique()) == 2:
        if recoded.unique()[0] < recoded.unique()[1]:
            # transform smaller number to zero
            recoded = recoded.replace(recoded.unique()[0], 0)
            # transform larger number to one
            recoded = recoded.replace(recoded.unique()[1], 1)
        else:
            # transform smaller number to zero
            recoded = recoded.replace(recoded.unique()[1], 0)
            # transform larger number to one
            recoded = recoded.replace(recoded.unique()[0], 1)
        new_dataframe['sex_encoded'] = recoded
    elif len(recoded.unique) < 2:
        print('there are at least two sexes,')
        print('your dataset appears to have fewer, caution, encode by hand')
    else:
        print('your dataset appears to have more than two sexes')
        print('caution, encode by hand,')

    return new_dataframe  # return dataframe with new column


def relate_columns_graphs(dataframe, special_column_name):
    """ This function makes a scatter plot of all columns

    :param dataframe: dataframe variable
    :type dataframe: pandas.dataFrame
    :param special_column_name: string of column you want to graph against
    :type  special_column_name: str

    :returns: no return, makes artifact
    :rtype: None.
    """
    y = dataframe[special_column_name].apply(pd.to_numeric)
    col = dataframe.columns.to_list()
    a = len(col)  # number of rows
    b = 1  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure(figsize=(10, (len(col)*10)))
    for i in col:
        plt.subplot(a, b, c)
        plt.scatter(dataframe[i].apply(pd.to_numeric), y)
        plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
        plt.xlabel(i)
        c = c + 1
    plt.savefig(("versus" + special_column_name + ".png"))


def relate_columns_graphs_two_dfs(
        dataframe1,
        dataframe2,
        special_column_name,
        other_column_name,
        color1='purple',
        color2='orange',
):

    """
    This function is meant to be a helper function
    for one that makes a scatter plot of all columns
    that two dataframes have in common

    :param dataframe1: dataframe variable
    :type dataframe1: pandas.dataFrame
    :param dataframe2: dataframe variable
    :type dataframe2: pandas.dataFrame
    :param special_column_name: str of column you graph against
    :type  special_column_name: str
    :param other_column_name: string of column you want to graph
    :type  other_column_name: str

    :returns: no return, makes artifact
    :rtype: None.
    """
    shared_columns = (
        dataframe1.columns.intersection(dataframe2.columns)).to_list()

    dataframe1 = dataframe1[shared_columns]
    dataframe2 = dataframe2[shared_columns]
    plt.scatter(
        dataframe1[special_column_name],
        dataframe1[other_column_name],
        color=color1,
        alpha=0.5,
    )
    plt.scatter(
        dataframe2[special_column_name],
        dataframe2[other_column_name],
        color=color2,
        alpha=0.5,
    )
    plt.xlabel(special_column_name)
    plt.ylabel(other_column_name)
    plt.savefig((other_column_name + "versus" + special_column_name + ".png"))
    plt.show(block=False)


def plot_2on2_df(dataframe1,
                 dataframe2,
                 special_column,
                 color1='purple',
                 color2='orange',):
    """
    This function is meant to create an artifact
    of two datasets with comparable variables
    in terms of graphing the variables
    against a variable of interest


    :param dataframe1: dataframe variable
    :type dataframe1: pandas.dataFrame
    :param dataframe2: dataframe variable
    :type dataframe2: pandas.dataFrame
    :param special_column_name: string of column you want to graph against
    :type  special_column_name: str

    :returns: no return, makes artifact
    :rtype: None.
    """
    shared_columns = (
        dataframe1.columns.intersection(dataframe2.columns)).to_list()
    for rotator_column in dataframe1[shared_columns]:
        relate_columns_graphs_two_dfs(
            dataframe1,
            dataframe2,
            special_column,
            rotator_column,
            color1=color1,
            color2=color2,
        )


def polyfit_and_show(
        dataframe,
        special_column_name,
        other_column_name,
        color1='purple',
):
    """
    This function creates a polynomial for two columns.
    It returns the coefficients in a 2nd degree polynomial
    and also creates a graph as a side effect.

    :param dataframe: dataframe variable
    :type dataframe: pandas.dataFrame
    :param special_column_name: string of column you want to graph against
    :type  special_column_name: str
    :param other_column_name: string of column you want to graph
    :type  other_column_name: str

    :returns: coeffiects
    :rtype: numpy.ndarray
    """
    x = np.array(dataframe[special_column_name])
    y = np.array(dataframe[other_column_name])
    degree = 2
    coefficients = np.polyfit(x, y, degree)
    print("Coëfficiënten 2nd degree polynomial:", coefficients)
    tup = (x.min(), x.max())
    line_z = []
    for a in tup:
        z = coefficients[0]*(a*a) + coefficients[1]*a + coefficients[2]
        line_z.append(z)
    plt.plot(tup, line_z)
    plt.scatter(
        dataframe[special_column_name],
        dataframe[other_column_name],
        color=color1,
        alpha=0.4,
    )
    return coefficients


def concat_double_header(dataframe_dub):
    """ This function concatenates the two headers of a dataframe
    :param dataframe_dub: dataframe with double header
    :type dataframe_dub: pandas.dataFrame


    :returns: dataframe with a single header
    :rtype: pandas.dataFrame
    """
    dataframe = dataframe_dub.copy()
    dataframe.columns = [c[0] + "_" + c[1] for c in dataframe.columns]
    return dataframe


def check_identical_columns(tsv_path):
    """
    Here we enter the path to a folder, then return the columns in
    which all files are exactly duplicated in name and values.
    """
    tsv_files = glob.glob(os.path.join(tsv_path, '*.tsv'))
    dataframes = [
        pd.read_csv(file, sep='\t', header=[0, 1], index_col=0)
        for file in tsv_files
    ]
    key_df, *rest_dfs = dataframes

    shared_columns = set(key_df.columns)

    for frame in rest_dfs:
        # check which labels are shared
        shared_columns = shared_columns.intersection(frame.columns)

    result = []
    for column in shared_columns:
        for frame in rest_dfs:
            if not frame[column].equals(key_df[column]):
                break
        else:
            result.append(column)
    return result
