
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and VUMC(?).
Licensed under <TBA>. See LICENSE for details.

This file contains functions for processing csv and tsv
files towards correct formats.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


def recode_sex(whole_dataframe, string_for_sex):
    """
    This function recodes sex into a new column if there
    are two possible values. It maintains numerical older
    butchanges the values to 0 and 1. The new column is
    called 'sex_encoded'.

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
    if len(dataframe_column.unique()) == 2:
        if recoded.unique()[0] < recoded.unique()[1]:
            # transform smaller number to zero
            recoded = recoded.replace(recoded.unique()[0], 0)
            recoded = recoded.replace(recoded.unique()[1], 1)
        else:
            recoded = recoded.replace(recoded.unique()[1], 0)
            recoded = recoded.replace(recoded.unique()[0], 1)
        new_dataframe['sex_encoded'] = recoded
    elif len(recoded.unique) < 2:
        print('there are two sexes, you have fewer, caution, encode by hand')
    else:
        print('there are two sexes, you have more, caution, encode by hand')

    return new_dataframe


def relate_columns_graphs(dataframe, special_column_name):
    """

    :param dataframe: dataframe variable
    :type dataframe: pandas.dataFrame
    :param special_column_name: string of column you want to graph against
    :type  special_column_name: str

    :returns: no return, makes artifact
    :rtype: None.
    """
    y = dataframe[special_column_name]
    col = dataframe.columns.to_list()
    a = len(col)  # number of rows
    b = 1  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure(figsize=(10, (len(col)*3)))
    for i in col:
        plt.subplot(a, b, c)
        plt.scatter(dataframe[i], y)
        plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
        plt.xlabel(i)
        c = c + 1
    plt.savefig(("versus" + special_column_name + ".png"))


def concat_double_header(dataframe_dub):
    """
    :param dataframe_dub: dataframe with double header
    :type dataframe_dub: pandas.dataFrame


    :returns: dataframe with a single header
    :rtype: pandas.dataFrame
    """
    dataframe = dataframe_dub.copy()
    dataframe.columns = [c[0] + "_" + c[1] for c in dataframe.columns]
    return dataframe
