"""
Copyright 2023 Netherlands eScience Center and Amsterdam University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains tests for the cvasl library.
"""
# sanity tests in docker for the  library


import unittest
import os
import subprocess
import pandas as pd
import numpy as np
import scipy
from tempfile import TemporaryDirectory
import json
from unittest import TestCase, main
from sklearn.linear_model import LinearRegression
# config
from cvasl.file_handler import Config
# hash_rash
from cvasl.file_handler import hash_rash
from cvasl.file_handler import intersect_all
# seperated
from cvasl.seperated import check_identical_columns
from cvasl.seperated import find_original_y_values
from cvasl.seperated import generate_transformation_matrix
from cvasl.seperated import find_outliers_by_list
from cvasl.seperated import check_sex_dimorph_expectations
from cvasl.seperated import bin_dataset
from cvasl.seperated import stratified_one_category_shuffle_split
from cvasl.seperated import stratified_cat_and_cont_categories_shuffle_split
from cvasl.seperated import pull_off_unnamed_column
from cvasl.seperated import recode_sex_to_numeric
# harmony
from cvasl.harmony import top_and_bottom_by_column
from cvasl.harmony import split_frame_half_balanced_by_column
from cvasl.harmony import log_out_columns


sample_test_data1 = os.path.join(
    os.path.dirname(__file__),
    '../test_data',
    'ar1.npy',
)

sample_test_data2 = os.path.join(
    os.path.dirname(__file__),
    '../test_data',
    'ar2.npy',
)



class TestConfig(TestCase):

    # required_directories = {
    #     'bids',
    #     'raw_data',
    #     'derivatives',
    # }
    required_directories = ['bids']

    def test_roots_only(self):
        with TemporaryDirectory() as td:
            bids = os.path.join(td, 'bids')
            os.mkdir(bids)
            raw_data = os.path.join(bids, 'raw_data')
            os.mkdir(raw_data)
            derivatives = os.path.join(bids, 'derivatives')
            os.mkdir(derivatives)
            raw_config = {
                'bids': bids,
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)

            config = Config.from_file(config_file)
            assert (
                config.get_directory('bids')
            )

    def test_missing_config_path(self):
        try:
            config = Config.from_file('')
        except ValueError:
            pass
        else:
            assert False, 'Didn\'t notify on missing config file'
             


class TestDockerTests(unittest.TestCase):

    def test_for_docker(self):
        p = np.load(sample_test_data1)
        a = np.load(sample_test_data2)
        self.assertEqual((p.sum() + 24),(a.sum()))

                            





if __name__ == '__main__':
    unittest.main()