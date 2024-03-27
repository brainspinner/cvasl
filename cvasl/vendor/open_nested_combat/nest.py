# The folling is minimally modified code from the library:
#  https://github.com/hannah-horng/opnested-combat

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
from ..neurocombat import neurocombat as nC


def OPNestedComBat(
        dat,
        covars,
        batch_list,
        filepath,
        categorical_cols=None,
        continuous_cols=None,
        return_estimates=False,
):
    """
    This function is from the Hannah Horng library open nested combat here :
    https://github.com/hannah-horng/opnested-combat
    As the library is unreleased and unversioned,
    we are using the MIT lisenced functions directly
    to version control them. There are some
    minimal changes for the sake of format correctness

    According to Dr. Horng's documentation this function
    "
    Completes sequential OPNested ComBat harmonization on an input DataFrame.
    Order is determined by running through all
    possible permutations of the order, then picking the order with the
    lowest number of features with significant
    differences in distribution."

    Arguments
    ---------
    dat : DataFrame of original data with shape (features, samples)
    covars : DataFrame with shape (samples, covariates)
    corresponding to original data.
    All variables should be label-
    encoded (i.e. strings converted to integer designations)
    batch_list : list of strings indicating batch effect column names
    within covars (i.e. ['Manufacturer', 'CE'...])
    filepath : root directory path for saving KS test p-values and
    kernel density plots created during harmonization
    categorical_cols : string or list of strings of categorical
    variables to adjust for
    continuous_cols : string or list of strings of continuous
    variables to adjust for
    return_estimates : if True, function will return both
    output_df and final_estimates

    Returns
    -------
    output_df : DataFrame with shape (features, samples) that
    has been sequentially harmonized with Nested ComBat
    final_estimates : list of dictionaries of estimates from
    iterative harmonization, used if user is deriving estimates
    from training data that need to be applied
    to a separate validation dataset

    """
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    perm_list = list(permutations(np.arange(len(batch_list))))
    count_dict = {}
    feature_dict = {}
    estimate_dict = {}
    c = 0
    for order in perm_list:
        c += 1
        n_dat = dat.copy()
        estimate_list = []
        print('Iteration ' + str(c) + ' of ' + str(len(perm_list)))
        for i in order:
            batch_col = batch_list[i]
            output = nC.neuroCombat(
                n_dat,
                covars,
                batch_col,
                continuous_cols=continuous_cols,
                categorical_cols=categorical_cols
            )
            output_df = pd.DataFrame.from_records(output['data'].T)
            n_dat = output_df.T
            estimate_list.append(output['estimates'])
        output_df.columns = dat.index
        feature_dict[str(order)] = n_dat
        count_dict[str(order)] = 0
        estimate_dict[str(order)] = estimate_list
        for batch_col in batch_list:
            p_list = []
            for j in range(len(output_df.columns)):
                feature = output_df.iloc[:, j]
                split_col = [
                    feature[covars[batch_col] == i]
                    for i in covars[batch_col].unique()
                ]
                p_list.append(anderson_ksamp(split_col).significance_level)
            count_dict[str(order)] += np.sum(np.asarray(p_list) < 0.05)
    if len(batch_list) != 1:
        best_order = [
            key
            for key, value in count_dict.items()
            if value == min(count_dict.values())
        ][0]
        best_order_list = list(map(int, best_order[1:-1].split(', ')))
        order = [batch_list[i] for i in best_order_list]
        n_dat = feature_dict[best_order]
        final_estimate = estimate_dict[best_order]

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
    This function is from the Hannah Horng library open nested combat here:
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
