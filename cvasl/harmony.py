
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
from sklearn.mixture import GaussianMixture
from scipy.stats import ranksums, ttest_ind
from scipy.stats import ttest_rel, ks_2samp, anderson_ksamp
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

# functions from opncombat


# def OPNestedComBat(
#         dat,
#         covars,
#         batch_list,
#         filepath,
#         categorical_cols=None,
#         continuous_cols=None,
#         return_estimates=False,
# ):
#     """
#     This function is from the Hannah Horng library open nested combat here :
#     https://github.com/hannah-horng/opnested-combat
#     As the library is unreleased and unversioned,
#     we are using the MIT lisenced functions directly
#     to version control them. There are some
#     minimal changes for the sake of format correctness

#     According to Dr. Horng's documentation this function

#     "Completes sequential OPNested ComBat harmonization
        # on an input DataFrame.
#     Order is determined by running through all
#     possible permutations of the order, then picking the order with the
#     lowest number of features with significant
#     differences in distribution."

#     Arguments
#     ---------
#     dat : DataFrame of original data with shape (features, samples)
#     covars : DataFrame with shape (samples, covariates)
#     corresponding to original data.
#     All variables should be label-
#         encoded (i.e. strings converted to integer designations)
#     batch_list : list of strings indicating batch effect column names
#       within covars (i.e. ['Manufacturer', 'CE'...])
#     filepath : root directory path for saving KS test p-values and
#       kernel density plots created during harmonization
#     categorical_cols : string or list of strings of categorical
#       variables to adjust for
#     continuous_cols : string or list of strings of continuous
#       variables to adjust for
#     return_estimates : if True, function will return both
#       output_df and final_estimates

#     Returns
#     -------
#     output_df : DataFrame with shape (features, samples) that
#       has been sequentially harmonized with Nested ComBat
#     final_estimates : list of dictionaries of estimates from
#       iterative harmonization, used if user is deriving estimates
#         from training data that need to be applied
#           to a separate validation dataset

#     """
#     if not os.path.exists(filepath):
#         os.makedirs(filepath)

#     perm_list = list(permutations(np.arange(len(batch_list))))
#     count_dict = {}
#     feature_dict = {}
#     estimate_dict = {}
#     c = 0
#     for order in perm_list:
#         c += 1
#         n_dat = dat.copy()
#         estimate_list = []
#         print('Iteration ' + str(c) + ' of ' + str(len(perm_list)))
#         for i in order:
#             batch_col = batch_list[i]
#             output = nC.neuroCombat(
#                 n_dat,
#                 covars,
#                 batch_col,
#                 continuous_cols=continuous_cols,
#                 categorical_cols=categorical_cols
#             )
#             output_df = pd.DataFrame.from_records(output['data'].T)
#             n_dat = output_df.T
#             estimate_list.append(output['estimates'])
#         output_df.columns = dat.index
#         feature_dict[str(order)] = n_dat
#         count_dict[str(order)] = 0
#         estimate_dict[str(order)] = estimate_list
#         for batch_col in batch_list:
#             p_list = []
#             for j in range(len(output_df.columns)):
#                 feature = output_df.iloc[:, j]
#                 split_col = [
#                     feature[covars[batch_col] == i]
#                     for i in covars[batch_col].unique()
#                 ]
#                 p_list.append(anderson_ksamp(split_col).significance_level)
#             count_dict[str(order)] += np.sum(np.asarray(p_list) < 0.05)
#     if len(batch_list) != 1:
#         best_order = [
#             key
#             for key, value in count_dict.items()
#             if value == min(count_dict.values())
#         ][0]
#         best_order_list = list(map(int, best_order[1:-1].split(', ')))
#         order = [batch_list[i] for i in best_order_list]
#         n_dat = feature_dict[best_order]
#         final_estimate = estimate_dict[best_order]

    print('Final Order: ' + str(order))

    txt_path = filepath + 'order.txt'
    with open(txt_path, 'w') as f:
        for item in order:
            f.write("%s\n" % item)

    output_df = pd.DataFrame.from_records(n_dat.T)
    output_df.columns = dat.index
    if return_estimates:
        return output_df, final_estimate
    else:
        return output_df


def feature_ad(dat, output_df, covars, batch_list, filepath):
    """
    This function is from the Hannah Horng library open nested combat here :
      https://github.com/hannah-horng/opnested-combat
    As the library is unreleased and unversioned,
    we are using the MIT lisenced functions directly
        to version control them. There are minimal
        changes for linting purposes.

    According to Dr. Horng's documentation this function
    "Computes AD test p-values separated by batch effect groups for a
    dataset (intended to assess differences in
    distribution to all batch effects in batch_list
    following harmonization NestedComBat"

    Arguments
    ---------
    dat : DataFrame of original data with shape (samples, features)
    output_df: DataFrame of harmonized data with shape (samples, features)
    covars : DataFrame with shape (samples, covariates) corresponding to
      original data. All variables should be label-
            encoded (i.e. strings converted to integer designations)
    batch_list : list of strings indicating batch effect column names
      within covars (i.e. ['Manufacturer', 'CE'...])
    filepath : write destination for kernel density plots and p-values

    If a feature is all the same value, the AD test cannot be completed.

    """
    p_df_original = pd.DataFrame()
    p_df_combat = pd.DataFrame()
    for batch_col in batch_list:

        # Computing KS Test P-Values
        p_list_original = []
        p_list_combat = []
        for j in range(len(output_df.columns)):
            feature_original = dat.iloc[:, j]
            feature_combat = output_df.iloc[:, j]
            try:
                split_col_original = [
                    feature_original[covars[batch_col] == i]
                    for i in covars[batch_col].unique()
                ]
                p_list_original.append(
                    anderson_ksamp(split_col_original).significance_level
                )
                split_col_combat = [
                    feature_combat[covars[batch_col] == i]
                    for i in covars[batch_col].unique()
                ]
                p_list_combat.append(
                    anderson_ksamp(split_col_combat).significance_level
                )
            except ValueError:
                print('Feature is all same value: ' + output_df.columns[j])

        p_df_original[batch_col] = p_list_original
        p_df_combat[batch_col] = p_list_combat

    p_df_original.index = dat.columns
    p_df_combat.index = output_df.columns
    p_df_original.to_csv(filepath + 'p_values_original.csv')
    p_df_combat.to_csv(filepath + 'p_values_combat.csv')


def GMMSplit(dat, caseno, filepath):
    """
    The following is from the Hannah Horng library open nested combat here :
    https://github.com/hannah-horng/opnested-combat
    As the library is unreleased and unversioned,
    we are using the MIT lisenced functions directly
    to version control them.

    According to Dr. Horng's documentation this function

    "Completes Gaussian Mixture model fitting and ComBat harmonization
    by the resulting sample grouping. The assumption
    here is that there is an unknown batch effect causing
    bimodality such that we can estimate the sample groupings for
    this hidden batch effect from the distribution. This function will
    take in a dataset, determine the best 2-component
    Gaussian mixture model, and use the resulting sample grouping to
    harmonize the data with ComBat." [needs better citation]

    Arguments
    ---------
    dat :
    DataFrame of original data with shape (features, samples)
    caseno :
    DataFrame/Series containing sample IDs
    (should be aligned with dat and covars),
    used to return sample
    grouping assignments.
    filepath :
    root directory path for saving the grouping and
    corresponding kernel density plots
    -------
    new_dat :
    DataFrame with shape (features, samples)
    that has been sequentially harmonized with Nested ComBat

    """
    # GENERATING GMM GROUPING
    data_keys = list(dat.T.keys())
    aic_values = []
    predictions = []
    col_list = []
    final_keys = []
    filepath2 = filepath+'GMM_Split/'
    if not os.path.exists(filepath2):
        os.makedirs(filepath2)

    for i in range(len(data_keys)):
        # print(col)
        feature = dat.T.iloc[:, i]
        X = pd.DataFrame({0: feature, 1: feature})
        gmix = GaussianMixture(n_components=2)
        col = data_keys[i]
        try:
            gmix.fit(X)
            results = gmix.predict(X)
            cluster_0 = X[results == 0].iloc[:, 0]
            cluster_1 = X[results == 1].iloc[:, 0]

            if (
                (len(cluster_0) <= .25*len(caseno)) or
                (len(cluster_1) <= .25*len(caseno))
            ):
                print('Clusters unbalanced: ' + data_keys[i])
            else:
                try:
                    plt.figure()
                    cluster_0.plot.kde()
                    cluster_1.plot.kde()
                    X.iloc[:, 0].plot.kde()
                    plt.legend(['Cluster 0', 'Cluster 1', 'Original'])
                    plt.xlabel(data_keys[i])
                    f = filepath2 + 'histogram_' + data_keys[i] + ".png"
                    filename = f   # hack for linting
                    plt.savefig(filename, bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    plt.close()
                    print('Failed to plot: ' + col, e)
                final_keys.append(col)
                predictions.append(results)
                aic_values.append(gmix.aic(X))
                col_list.append(col)
        except ValueError:
            print('Failed to fit: ' + col)

    # Returning AIC values
    gauss_df = pd.DataFrame({'Feature': final_keys, 'AIC': aic_values})
    best_fit = gauss_df[
        gauss_df['AIC'] == min(gauss_df['AIC'])]['Feature'].iloc[0].strip(' ')
    best_fit_n = gauss_df[
        gauss_df['AIC'] == min(gauss_df['AIC'])]['Feature'].index[0]
    gauss_df.to_csv(filepath2 + 'GaussianMixture_aic_values.csv')

    # Returning patient split
    predictions_df = pd.DataFrame()
    predictions_df['Patient'] = caseno
    predictions_df['Grouping'] = predictions[best_fit_n]
    predictions_df.to_csv(filepath2 + best_fit + '_split.csv')

    return predictions_df
