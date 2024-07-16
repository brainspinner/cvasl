# harm2


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
import numpy as np
from neuroHarmonize import harmonizationLearn


def neuroharmony_apply(
        stuck_together_data,
        list_of_covariates,
        list_features_to_harmonize,
):
    stuck_together_datasets_cov = stuck_together_data[list_of_covariates]
    covariates = stuck_together_datasets_cov.rename(
        columns={'Site': 'SITE'})
    # should expand to accept siteor get rid of?
    harmonizable_features = stuck_together_data[list_features_to_harmonize]
    features_array = np.array(harmonizable_features)
    my_model, my_data_adj = harmonizationLearn(features_array, covariates)
    neuroharmonized_ = pd.DataFrame(
        my_data_adj,
        columns=list_features_to_harmonize)

    neuroharmonized_ = pd.concat(
        [neuroharmonized_, stuck_together_datasets_cov.reset_index()],
        axis=1)
    neuroharmonized_ = neuroharmonized_.drop('index', axis=1)
    return neuroharmonized_, stuck_together_datasets_cov, features_array
