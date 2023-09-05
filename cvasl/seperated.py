
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and Stichting VUMC.
Licensed under <TBA>. See LICENSE for details.

This file contains functions for processing csv and tsv
files towards correct formats.
"""

import pandas as pd
from pandas.api.types import is_numeric_dtype
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


def relate_columns_graphs(dataframe, special_column_name, saver=False):
    """ This function makes a scatter plot of all columns

    :param dataframe: dataframe variable
    :type dataframe: pandas.dataFrame
    :param special_column_name: string of column you want to graph against
    :type  special_column_name: str
    :param saver: bool to indicate if graph pngs should be saved
    :type saver: bool

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
    if saver is True:
        plt.savefig(("versus" + special_column_name + ".png"))


def relate_columns_graphs_numeric(
        dataframe,
        special_column_name,
        saver=False,
):
    """
    This function makes a scatter plot of all columns that are numeric.

    :param dataframe: dataframe variable
    :type dataframe: pandas.dataFrame
    :param special_column_name: string of column you want to graph against
    :type  special_column_name: str
    :param saver: string to indicate if graph pngs should be saved
    :type saver: str

    :returns: no return, makes artifact
    :rtype: None.
    """
    y = dataframe[special_column_name].apply(pd.to_numeric)
    col = dataframe.columns.to_list()
    a = len(col)  # number of rows
    b = 1  # number of columns
    c = 1  # initialize plot counter
    fig = plt.figure(figsize=(4, (len(col)*4)))
    for i in col:
        if is_numeric_dtype(dataframe[i]):
            plt.subplot(a, b, c)
            plt.scatter(dataframe[i], y)
            plt.title('{}, subplot: {}{}{}'.format(i, a, b, c))
            plt.xlabel(i)
            plt.ylabel(special_column_name)
        c = c + 1
    if saver is True:
        plt.savefig(("versus" + special_column_name + ".png"))
    else:
        pass


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
        degree_poly,
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
    :type other_column_name: str
    :param degree_poly: either 1,2 or 3 only
    :type  degree_poly: int
    :param color1: string of color for graphing
    :type color1: str

    :returns: coeffiects
    :rtype: :class:`~numpy.ndarray`
    """
    dataframe = dataframe.dropna()
    xscat = np.array(pd.to_numeric(dataframe[special_column_name]))
    yscat = np.array(pd.to_numeric(dataframe[other_column_name]))
    coefficients = np.polyfit(xscat, yscat, degree_poly)
    print("Coefficents for", degree_poly, "degree polynomial:", coefficients)
    x = np.array((range(int(xscat.max()))))
    if degree_poly == 2:
        y = coefficients[0]*(x**2) + coefficients[1]*x + coefficients[2]
    elif degree_poly == 1:
        y = coefficients[0]*x + coefficients[1]
    elif degree_poly == 3:
        y = (
                coefficients[0]*(x**3) +
                coefficients[1]*(x**2) +
                coefficients[2]*x +
                coefficients[3]
            )
    else:
        print("Function does not support degree over 3, and should fail")

    plt.plot(x, y)
    plt.scatter(
        xscat,
        yscat,
        color=color1,
        alpha=0.4,
    )
    return coefficients


def concat_double_header(dataframe_dub):
    """
    This function concatenates the two headers of a dataframe
    :param dataframe_dub: dataframe with double header
    :type dataframe_dub: pandas.dataFrame


    :returns: dataframe with a single header
    :rtype: pandas.dataFrame
    """
    dataframe = dataframe_dub.copy()
    dataframe.columns = [c[0] + "_" + c[1] for c in dataframe.columns]
    return dataframe


def check_identical_columns(tsv_path, header=0):
    """
    Here we enter the path to a folder, then return the columns in
    which all files are exactly duplicated in name and values.

    needs more
    """
    tsv_files = glob.glob(os.path.join(tsv_path, '*.tsv'))
    if header == 1:
        dataframes = [
            pd.read_csv(file, sep='\t', header=[0, 1], index_col=0)
            for file in tsv_files
        ]
    else:
        dataframes = [
            pd.read_csv(file, sep='\t', header=[0], index_col=0)
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


def generate_transformation_matrix(polynomial1, polynomial2):
    """
    Generates a matrix that transforms one polynomial into another.
    :param polynomial1: coefficients of the polynomial in the form (a1, b1)
    :type polynomial1: tuple
    :param polynomial2: coefficients of the polynomial in the form (a2, b2)
    :type polynomial2: tuple


    :returns: m, an array
    :rtype: ~numpy.ndarrray
    """

    if len(polynomial1) != len(polynomial2):
        raise ValueError('Polynomials must be of equal size.')

    m = np.zeros([len(polynomial1), len(polynomial1)])
    for i, (c1, c2) in enumerate(zip(polynomial1, polynomial2)):
        m[i][i] = c2/c1

    return m


def find_original_y_values(polynomial, output_value):
    """
    Finds the original y-values of a second or third degree polynomial
    given its coefficients and an output value.

    :param polynomial: coefficients of the polynomial in the form (a, b, c)
    :type polynomial: tuple

    :param output_value: output of polynomial when a list of y are given
    :type output_value: list

    :returns: pile,list of original y-values corresponding to the output value
    :rtype: list
    """

    pile = []

    if len(polynomial) == 3:
        a, b, c = polynomial

        for value in output_value:
            # calculate the discriminant
            discriminant = b**2 - 4*a*(c - value)
            # if the discriminant is negative, no real roots exist
            if discriminant < 0:
                return []
            # calculate the original y-values
            x1 = (-b + np.sqrt(discriminant)) / (2*a)
            pile.append(x1)

    elif len(polynomial) == 2:
        a, b = polynomial
        for value in output_value:
            # calculate the original y-values
            x1 = (value - b)/a
            pile.append(x1)

    else:
        raise NotImplementedError(
            'find_original_y_values only implemented for second \
                or third degree polynomials.'
        )

    return pile


def find_outliers_by_list(dataframe, column_list, number_sd):
    """
    This function finds the outliers in terms of anything outside
    a given number of
    standard deviations (number_sd)
    from the mean on a list of specific specific column,
    then returns these rows of the dataframe.

    :param dataframe: whole dataframe on dataset
    :type dataframe: ~pandas.DataFrame
    :param column_list: list of relevant columns
    :type column_list: list
    :param number_sd: number of standard deviations
    :type number_sd: float

    :returns: dataframe of outliers
    :rtype: ~pandas.DataFrame
    """
    outlier_frames = []
    for column_n in column_list:
        mean = dataframe[column_n].mean()
        std = dataframe[column_n].std()
        values = dataframe[column_n].abs() - abs(mean + (number_sd * std))
        outliers = dataframe[values > 0]
        outlier_frames.append(outliers)
    outlier_super = pd.concat(outlier_frames)
    outlier_super = outlier_super.loc[~outlier_super.duplicated()].copy()
    return outlier_super


def check_sex_dimorph_expectations(dataframe):
    """
    This function checks that men
    as expected have larger brains than women
    in a given dataframe.

    :param dataframe: dataframe with cvasl standard for patient MRI data
    :type dataframe: ~pandas.DataFrame

    :returns: dataframe, or zero, with side effect of printed information
    :rtype: ~pandas.DataFrame or int
    """
    ladies = dataframe[dataframe['sex'] == 'F']
    men = dataframe[dataframe['sex'] == 'M']
    print('You have', len(ladies)/len(men), 'times as many ladies than men')
    if ladies.gm_vol.mean() < men.gm_vol.mean():
        print('As expected men have larger grey matter')
    if ladies.wm_vol.mean() < men.wm_vol.mean():
        print('As expected men have larger white matter')
    if ladies.gm_vol.mean() >= men.gm_vol.mean():
        print(
            'Caution, average female grey matter may be \
                  at similar or larger size than men'
        )
    if ladies.wm_vol.mean() >= men.wm_vol.mean():
        print(
            'Caution, average female white matter may be \
                  at similar or larger size than men'
        )
    if ladies.gm_vol.mean() >= men.gm_vol.mean() \
            or ladies.wm_vol.mean() >= men.wm_vol.mean():
        bad_data = dataframe
    else:
        bad_data = 0
    return bad_data
