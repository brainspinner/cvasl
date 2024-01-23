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
