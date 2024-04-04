
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and
the Amsterdam University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for processing csv and tsv
files towards correct formats.
"""

import os
import sys
import warnings
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import copy
import glob
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import mean_absolute_error

from .harmony import log_out_columns


def bin_dataset(dataframe, column, num_bins=4, graph=False):
    """
    This function creates an additional column
    where a continues variable can be binned
    into 2 or  4 parts.

    :param dataframe: dataframe variable
    :type dataframe: str
    :param column: column name written in singe qoutes
    :type column: str
    :param num_bins: 2 or 4 for number bins
    :type num_bins: int
    :param graph: on True setting produces split graph
    :type graph: bool

    :returns: dataframe with additional column
    :rtype: pandas.dataFrame
    """
    if num_bins == 2:
        bins = [
            dataframe[column].describe()['min'],
            dataframe[column].describe()['50%'],
            dataframe[column].describe()['max']]
        labels = [1, 2]
        dataframe['binned'] = pd.cut(dataframe[column], bins, labels=labels)
    else:
        bins = [
            dataframe[column].describe()['min'] - 1,
            dataframe[column].describe()['25%'],
            dataframe[column].describe()['50%'],
            dataframe[column].describe()['75%'],
            dataframe[column].describe()['max'] + 1]
        labels = [0, 1, 2, 3]
        dataframe['binned'] = pd.cut(dataframe[column], bins, labels=labels)
    if num_bins != 2 and num_bins != 4:
        print("You can only bin into 2 or 4 bins, we defaulted to 4 for you")
    if graph:
        sns.displot(dataframe, x=column, hue='binned')
    return dataframe


def static_bin_age(dataframe):
    """
    This function applies static binning by age decade
    to a dataframe
    """
    dataframe['static_bin_age'] = 0
    dataframe.loc[dataframe['age'] >= 10, 'static_bin_age'] = 1
    dataframe.loc[dataframe['age'] >= 20, 'static_bin_age'] = 2
    dataframe.loc[dataframe['age'] >= 30, 'static_bin_age'] = 3
    dataframe.loc[dataframe['age'] >= 40, 'static_bin_age'] = 4
    dataframe.loc[dataframe['age'] >= 50, 'static_bin_age'] = 5
    dataframe.loc[dataframe['age'] >= 60, 'static_bin_age'] = 6
    dataframe.loc[dataframe['age'] >= 70, 'static_bin_age'] = 7
    dataframe.loc[dataframe['age'] >= 80, 'static_bin_age'] = 8
    dataframe.loc[dataframe['age'] >= 90, 'static_bin_age'] = 9
    dataframe.loc[dataframe['age'] >= 100, 'static_bin_age'] = 10
    return dataframe


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


def pull_off_unnamed_column(unclean, extra_columns=[]):
    """
    This function takes a dataframe
    and if there are columns with the string
    "Unnamed" it drops them.
    It also drops the extra columns you input
    """
    df = unclean[unclean.columns.drop(list(unclean.filter(regex='Unnamed')))]
    df = df.drop(extra_columns, axis=1)
    return df


def drop_columns_folder(directory, list_droppables):
    """
    This function works  csvs in a folder
    on those with unnamed columns and
    other unwanted columns
    it drops them, they are then available
    in a new folder called 'stripped'

    :param directory: directory where csv are variable
    :type directory: str

    :returns: dataframes without unnamed columns
    :rtype: list

    """
    collection = []
    directory_list = glob.glob(directory + "/*.csv")
    for frame in directory_list:
        framed = pd.read_csv(frame,)
        output = pull_off_unnamed_column(framed)
        output = output.drop(list_droppables, axis=1)
        recoded_dir = 'stripped'
        if not os.path.exists(recoded_dir):
            os.makedirs(recoded_dir)
        frame_s = os.path.split(frame)
        name = ('stripped/' + frame_s[-1][:-4] + 'stripped.csv')
        output.to_csv(name)

    return collection


def folder_chain_out_columns(datasets_folder, columns, output_folder):
    """
    This function works  csvs in a folder at any folder level inside
    on those with  unwanted columns
    it drops them, they are then available
    in a new folder called specified

    :param datasets_folder: directory where csv are variable
    :type  datasets_folder: str
    :param columns: list of columns as strings
    :type  columns: list
    :param output_folder: directory where newly made csvs are sent
    :type  output_folder: str

    :returns: None
    :rtype: None
    """
    directory_list = glob.glob(
        os.path.join(datasets_folder, '**/*.csv'), recursive=True
    )
    for frame in directory_list:
        # print(frame)
        framed = pd.read_csv(frame)
        output = framed.drop(columns, axis=1)
        relpath = os.path.dirname(frame)
        subpath = os.path.relpath(relpath, datasets_folder)
        parent = os.path.join(output_folder, subpath)
        try:
            # print('creating parent:', parent)
            os.makedirs(parent)
        except FileExistsError:
            pass
        output.to_csv(os.path.join(parent, os.path.basename(frame)))
        print('created file:', os.path.join(parent, os.path.basename(frame)))


def recode_sex_folder(directory):
    """
    This function recodes sex on csvs
    with such a column in a sepcified directory
    into csvs with a new column if there
    are two possible values. It maintains numerical order
    but changes the values to 0 and 1. The column is
    called changed. Note sex should be encoded
    in numbers i.e. ints for many functions.
    The new files are produced as a side effect.

    :param directory: directory where csv are variable
    :type directory: str

    :returns: dataframes with sex encoded correctly
    :rtype: list

    """
    collection = []
    directory_list = glob.glob(directory + "/*.csv")
    for frame in directory_list:
        framed = pd.read_csv(frame,)
        sex_mapping = {'F': 0, 'M': 1}
        output = framed.assign(sex=framed.sex.map(sex_mapping))
        recoded_dir = 'recoded'
        if not os.path.exists(recoded_dir):
            os.makedirs(recoded_dir)
        frame_s = os.path.split(frame)
        name = ('recoded/' + frame_s[-1][:-4] + 'recoded.csv')
        print(frame_s[-1])
        output.to_csv(name)

    return collection


def make_log_file(file_name, list_of_columns):
    """
    This function recodes columns on a csv
    file into their log value

    :param file_name:  csv with  variables as columns
    :type file_name: str

    :returns: dataframe
    :rtype: ~pandas.dataframe

    """

    framed = pd.read_csv(file_name)
    output = log_out_columns(framed, list_of_columns)
    recoded_dir = 'loged_file'
    if not os.path.exists(recoded_dir):
        os.makedirs(recoded_dir)
    frame_s = os.path.split(file_name)
    name = ('loged_file/' + frame_s[-1][:-4] + 'loged.csv')
    output.to_csv(name)

    return output


def make_log_folder(directory, list_of_columns):
    """
    This function recodes columns on csvs
    with such a column in a sepcified directory
    into csvs with a new column is the log of old.
    The new files are produced as a side effect.

    :param directory: directory where csv are variable
    :type directory: str

    :returns: dataframes with sex encoded correctly
    :rtype: list

    """
    collection = []
    directory_list = glob.glob(directory + "/*.csv")
    for frame in directory_list:
        framed = pd.read_csv(frame,)
        output = log_out_columns(framed, list_of_columns)
        recoded_dir = 'loged'
        if not os.path.exists(recoded_dir):
            os.makedirs(recoded_dir)
        frame_s = os.path.split(frame)
        name = ('loged/' + frame_s[-1][:-4] + 'loged.csv')
        output.to_csv(name)

    return collection


def recode_sex_to_numeric(df):
    """When we need to flip the sex back to numbers from the
    suggested format this function will turn females to 1, males to 0"""
    sex_mapping = {'F': 1, 'M': 0}
    df = df.assign(sex=df.sex.map(sex_mapping))
    return df


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


def drop_y(df):
    """
    This is meant as a psuedo-helper function for pandas columns
    when they are merged. It drops columns that end in y
    """
    to_drop = [x for x in df if x.endswith('_y')]
    df.drop(to_drop, axis=1, inplace=True)
    return df


def polyfit_second_degree_to_df(
        dataframe,
        special_column_name,
        other_column_names,
):
    """
    This function creates polynomials for columns,
    as compared to a special column, in our case age.
    It returns the coefficients in a dataframe.

    :param dataframe: dataframe variable
    :type dataframe: pandas.dataFrame
    :param special_column_name: column name, usually age
    :type  special_column_name: str
    :param other_column_names: columns you want to get poly coefficients on
    :type other_column_names: list


    :returns: coeffiects
    :rtype: :class:`~numpy.ndarray`
    """
    list_as = []
    list_bs = []
    list_cs = []
    list_columns = []
    dataframe = dataframe.dropna()
    for interest_column_name in other_column_names:
        xscat = np.array(pd.to_numeric(dataframe[special_column_name]))
        yscat = np.array(pd.to_numeric(dataframe[interest_column_name]))
        coefficients = np.polyfit(xscat, yscat, 2)  # 2 = degree_poly
        list_columns.append(interest_column_name)
        list_as.append(coefficients[0])
        list_bs.append(coefficients[1])
        list_cs.append(coefficients[2])
    d = {
        'column': list_columns,
        'coefficient_a': list_as,
        'coefficient_b': list_bs,
        'coefficient_c': list_cs,
        }
    coefficient_dataframe = pd.DataFrame(d)

    return coefficient_dataframe


def derived_function(column, a, b, c):
    """
    This functions allows you to derive a projected
    value for any parameter based on a polynomial
    for age versus the parameter, given that
    your data is in a dataframe format.


    :param column: pandas dataframe variable column
    :type column: pandas.core.series.Series
    :param a: first coeffiecnt
    :type  a: float
    :param b: second coefficient
    :type  b: float
    :param c: final term in polynomial
    :type  c: float


    :returns: series
    :rtype: :class:`~pandas.core.series.Series`
    """
    return a * (column**2) + b * column + c


def concat_double_header(dataframe_dub):
    """
    This function concatenates the two headers of a dataframe
    :param dataframe_dub: dataframe with double header
    :type dataframe_dub: pandas.dataFrame


    :returns: dataframe with a single header
    :rtype: `~pandas.DataFrame`
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
    :param polynomial1: coefficients of the polynomial in form (a1, b1, ...)
    :type polynomial1: Sequence
    :param polynomial2: coefficients of the polynomial in form (a2, b2, ...)
    :type polynomial2: Sequence


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


def avg_k_folds(frame):
    """
    This function takes a dataframe of k_fold results,
    as formatted for our experiments with derived datasets
    and returns an averaged dataframe
    """
    data = [[
        frame['algorithm'][0],
        frame['file_name'][0],
        frame.mae.mean(),
        frame.r2.mean(),
        frame.explained_variance.mean()]]
    frame_results_average = pd.DataFrame(
        data,
        columns=['algorithm', 'file_name', 'mae', 'r2', 'explained_variance']
    )
    return frame_results_average


def stratified_one_category_shuffle_split(
        model_name,
        model_file_name,
        scikit_model,
        our_ml_matrix,
        our_x,
        our_y,
        category='sex',
        splits=5,
        test_size_p=0.20,
        printed=False
):
    """
    This takes a sci-kit learn coded model and
    creates a dataframe based on k-folds of results on
    our_ml_matrix, and it's X component
    returns a dataframe of fold results
    and raw y_test versus y_pred
    as well as a tuple with models
    and then the training data from the model.
    The random state in the StratifiedShuffleSplit is set, so
    the results should be reproducible.

    :param model_name: name of model
    :type model_name: str
    :param model_file_name: name offile where specific model will be stored
    :type model_file_name: str
    :param skikit_model: name of skikit-model
    :type skikit_model: str
    :param our_ml_matrix: dataframe to work over
    :type our_ml_matrix: `~pd.DataFrame`
    :param our_x: X or features columnfor machine learning
    :type our_x: dataframe
    :param our_y: y or label column for machine learning
    :type our_y: class:`~pandas.core.series.Series`
    :param category: categorical variable (column) to be stratified on eg. sex
    :type category: str
    :param splits: number of folds desired
    :type splits: int
    :param test_size_p: percent to put into test
    :type test_size_p: float
    :param printed: printed information on folds option
    :type printed: bool


    :returns: dataframe, y dataframe, and models
    :rtype: tuple
    """
    y_split = our_ml_matrix[category].values
    sss = StratifiedShuffleSplit(
        n_splits=splits,
        test_size=test_size_p,
        random_state=12
    )

    X = our_x
    # TODO: (makeda)finish split and put back index so everything is traceable
    y = our_y
    sss.get_n_splits(X, y_split)

    unique, counts = np.unique(y_split, return_counts=True)

    y_frame = []
    all_mod_results = []
    models = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y_split)):
        unique, counts = np.unique(y_split[train_index], return_counts=True)
        unique, counts = np.unique(y_split[test_index], return_counts=True)
        cols = [
            'algorithm',
            'fold',
            'file_name',
            'mae',
            'r2',
            'explained_variance',
        ]
        mod_results = pd.DataFrame(columns=cols)
        current_fold_X_train = X[train_index][:, 1:]
        current_fold_y_train = y[train_index]
        current_fold_X_test = X[test_index][:, 1:]
        current_fold_y_test = y[test_index]
        scikit_model.fit(current_fold_X_train, current_fold_y_train)
        current_fold_y_pred = scikit_model.predict(current_fold_X_test)
        if printed:
            print(f"\nFold {i}:")
            print(
                f'Train shapes: X {X[train_index].shape}',
                f' y {y[train_index].shape}'
            )
            unique_train, counts_train = np.unique(
                y_split[train_index], return_counts=True
            )
            print(
                f'Category classes: {unique_train}',
                f'percentages: {100*counts_train/y[train_index].shape[0]}'
            )
            print(
                f'\nTest shapes: X {X[test_index].shape}',
                f'  y {y[test_index].shape}'
            )
            unique_test, counts_test = np.unique(
                y_split[test_index], return_counts=True
            )
            print(
                f'Category classes: {unique_test},'
                f'percentages: {100*counts_test/y[test_index].shape[0]}'
            )

        data = [[
            f'{model_name}-{i}',
            i,
            f'{model_file_name}.{i}',
            mean_absolute_error(current_fold_y_test, current_fold_y_pred),
            scikit_model.score(current_fold_X_test, current_fold_y_test),
            metrics.explained_variance_score(
                current_fold_y_test,
                current_fold_y_pred
            )]]
        mod_results_current_fold = pd.DataFrame(data, columns=cols)
        mod_results = pd.concat([mod_results, mod_results_current_fold])
        mod_results.reset_index(drop=True, inplace=True)
        all_mod_results.append(mod_results)
        y_frame_now = pd.DataFrame(
            {
                'y_test': list(current_fold_y_test),
                'y_pred': list(current_fold_y_pred),
            })

        y_frame.append(y_frame_now)

        models.append((scikit_model, X[train_index][:, 0]))

    df = pd.concat(all_mod_results)
    y_frame = pd.concat([
        y_frame[0],
        y_frame[1],
        y_frame[2],
        y_frame[3],
        y_frame[4],
    ], axis=0)

    return df, y_frame, models


def stratified_cat_and_cont_categories_shuffle_split(
        model_name,
        model_file_name,
        scikit_model,
        our_ml_matrix,
        our_x,
        our_y,
        cat_category='sex',
        cont_category='age',
        splits=5,
        test_size_p=0.2,
        printed=False
):
    """
    This takes a sci-kit learn coded model and
    creates a dataframe based on (stratified) k-folds of results on
    our_ml_matrix, and it's X component
    returns a dataframe of fold results
    and raw y_test versus y_pred
    as well as a tuple with models
    and then the training data from the model.
    This is a twist on Stratified Shuffle Split
    to allow it's stratification on a categorical
    and continous variable. Note that the categorical
    should already be converted into integers before
    this function is run.
    The random state in the StratifiedShuffleSplit is set, so
    the results should be reproducible.

    :param model_name: name of model
    :type model_name: str
    :param model_file_name: name offile where specific model will be stored
    :type model_file_name: str
    :param skikit_model: name of skikit-model
    :type skikit_model: str
    :param our_ml_matrix: dataframe to work over
    :type our_ml_matrix: `~pd.DataFrame`
    :param our_x: X or features columnfor machine learning
    :type our_x: dataframe
    :param our_y: y or label column for machine learning
    :type our_y: class:`~pandas.core.series.Series`
    :param cat_category: categorical variable column to stratify on eg. sex
    :type cat_category: str
    :param cont_category: continuuous variable column to stratify on eg. age
    :type cont_category: str
    :param splits: number of folds desired
    :type splits: int
    :param test_size_p: percent to put into test
    :type test_size_p: float
    :param printed: printed information on folds option
    :type printed: bool


    :returns: dataframe, y dataframe, and models
    :rtype: tuple
    """
    if test_size_p > 1 / splits:
        message1 = "You us a potentially problematic percent (may resample) "
        warnings.warn(message1)
    our_ml_matrix = bin_dataset(
        our_ml_matrix,
        cont_category,
        num_bins=4,
        graph=False
    )
    our_ml_matrix['fuse_bin'] = (
        our_ml_matrix[cat_category] * len(
            our_ml_matrix['binned'].unique()
        ) + pd.to_numeric(our_ml_matrix['binned']))
    y_split = our_ml_matrix['fuse_bin'].values
    sss = StratifiedShuffleSplit(
        n_splits=splits,
        test_size=test_size_p,
        random_state=12
    )

    X = our_x
    # TODO: (makeda)finish split and put back index so everything is traceable
    y = our_y
    sss.get_n_splits(X, y_split)

    unique, counts = np.unique(y_split, return_counts=True)

    y_frame = []
    all_mod_results = []
    models = []
    for i, (train_index, test_index) in enumerate(sss.split(X, y_split)):
        unique, counts = np.unique(y_split[train_index], return_counts=True)
        unique, counts = np.unique(y_split[test_index], return_counts=True)
        cols = [
            'algorithm',
            'fold',
            'file_name',
            'mae',
            'r2',
            'explained_variance',
        ]
        mod_results = pd.DataFrame(columns=cols)
        current_fold_X_train = X[train_index][:, 1:]
        current_fold_y_train = y[train_index]
        current_fold_X_test = X[test_index][:, 1:]
        current_fold_y_test = y[test_index]
        scikit_model.fit(current_fold_X_train, current_fold_y_train)
        current_fold_y_pred = scikit_model.predict(current_fold_X_test)
        if printed:
            print(f"\nFold {i}:")
            print(
                f'Train shapes: X {X[train_index].shape}',
                f' y {y[train_index].shape}'
            )
            unique_train, counts_train = np.unique(
                y_split[train_index], return_counts=True
            )
            bins = our_ml_matrix['binned']
            # print(
            #     f'Category classes: {unique_train}',
            #     f'from categorical: {our_ml_matrix[cat_category].unique()} ',
            #     f'and continous binned to: {bins.unique()} ',
            #     f'percentages: {100*counts_train/y[train_index].shape[0]}'
            #     # TODO: shape[iterates- i to fold]?
            # )
            print(
                f'\nTest shapes: X {X[test_index].shape}',
                f'  y {y[test_index].shape}'
            )
            unique_test, counts_test = np.unique(
                y_split[test_index], return_counts=True
            )
            # print(
            #     f'Category classes: {unique_test},'
            #     f'percentages: {100*counts_test/y[test_index].shape[0]}'
            #     # TODO: shape[iterates i to fold]?
            # )

        data = [[
            f'{model_name}-{i}',
            i,
            f'{model_file_name}.{i}',
            mean_absolute_error(current_fold_y_test, current_fold_y_pred),
            scikit_model.score(current_fold_X_test, current_fold_y_test),
            metrics.explained_variance_score(
                current_fold_y_test,
                current_fold_y_pred
            )]]
        mod_results_current_fold = pd.DataFrame(data, columns=cols)
        mod_results = pd.concat([mod_results, mod_results_current_fold])
        mod_results.reset_index(drop=True, inplace=True)
        all_mod_results.append(mod_results)
        y_frame_now = pd.DataFrame(
            {
                'y_test': list(current_fold_y_test),
                'y_pred': list(current_fold_y_pred),
            })

        y_frame.append(y_frame_now)

        models.append((scikit_model, X[train_index][:, 0]))

    df = pd.concat(all_mod_results)
    y_frame = pd.concat([
        y_frame[0],
        y_frame[1],
        y_frame[2],
        y_frame[3],
        y_frame[4],
    ], axis=0)

    return df, y_frame, models


def preprocess(
    folder,
    file_extension,
    outcome_folder,
    log_cols=[],
    plus_one_log_columns=[],
):
    """
    This function given a directory will
    search all subdirectory for noted file extension
    Copies of the files will be processed as specified
    which is the specified columns turned to log or +1 then log
    then put in the outcome folder
    """
    if not os.path.exists(outcome_folder):
        os.makedirs(outcome_folder)
    files = '**/*.' + file_extension
    suspects = glob.glob(
        os.path.join(folder, files),
        recursive=True,
    )
    read_names = []
    for file in suspects:
        read = pd.read_csv(file, index_col=0)
        filenames1 = os.path.split(file)[0]
        filenames = os.path.split(filenames1)[-1]
        if not os.path.exists(os.path.join(outcome_folder, filenames)):
            os.makedirs(os.path.join(outcome_folder, filenames))
        filey = os.path.basename(file).split('/')[-1]
        read_name = os.path.join(outcome_folder, filenames, filey)
        read[plus_one_log_columns] = read[plus_one_log_columns].apply(
            lambda x: x + 1, axis=1)
        read[plus_one_log_columns] = read[plus_one_log_columns].apply(
            lambda x: np.log(x), axis=1)
        read[log_cols] = read[log_cols].apply(lambda x: np.log(x), axis=1)
        read.to_csv(read_name)
        read_names.append(read_name)
    return read_names
