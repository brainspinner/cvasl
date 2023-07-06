#sanity tests for the  library


import unittest
import os
import numpy as np
import scipy
from tempfile import TemporaryDirectory
import json
from unittest import TestCase, main

# config
from cvasl.file_handler import Config
# hash_rash
from cvasl.file_handler import hash_rash
from cvasl.file_handler import intersect_all
# seperated
from cvasl.seperated import check_identical_columns
from cvasl.seperated import find_original_y_values_quadratic


sample_test_data1 = os.path.join(
    '/cvasl/test_data',
    'ar1.npy',
)

sample_test_data2 = os.path.join(
    '/cvasl/test_data',
    'ar2.npy',
)


class TestConfig(TestCase):

    # required_directories = {
    #     'bids',
    #     'raw_data',
    #     'derivatives',
    # }
    required_directories = ['bids', 'raw_data', 'derivatives']

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

            config = Config(config_file)
            assert (
                config.get_directory('bids') and
                config.get_directory('raw_data') and
                config.get_directory('derivatives')
            )

    def test_missing_config_path(self):
        try:
            Config('non existent')
        except ValueError:
            pass
        else:
            assert False, 'Didn\'t notify on missing config file'
    

class TestHashMethods(unittest.TestCase):

    def test_hash_rash(self):
        tempfile1 = 'sample_mri_t.npy' # made up mri
        tempfile2 = 'sample_mri_t.npy' # another made up mri
        with TemporaryDirectory() as td:
            with open(os.path.join(td, tempfile1), 'w') as tf:
                tf.write('string')
            with open(os.path.join(td, tempfile2), 'w') as tf:
                tf.write('another_string')
            
            self.assertTrue((hash_rash(td, 'npy')["hash"]).equals(hash_rash(td, 'npy')["hash"]))


class TestPolynomiaMethods(unittest.TestCase):

    def test_find_original_y_values_quadratic(self):
        polynomial = 4,3,1
        original_values = [1,2,3,4,5,6]
        results_poly1 = []
        for xs in original_values:
            y = (4*xs**2 + xs*3 +1)
            results_poly1.append(y)
        mapped_back_values =find_original_y_values_quadratic(polynomial, results_poly1)
        print(type(original_values))
        print(type(mapped_back_values))         
        self.assertEqual((sum(original_values)),(sum(mapped_back_values)))

class TestSeperatedMethods(unittest.TestCase):
    #TODO: replace with test that runs over files in docker subdirectory

    # def test_check_identical_columns(self):
    #     tempfile1 = 'sample_mri_t.npy' # made up mri
    #     tempfile2 = 'sample_mri_t.npy' # another made up mri
    #     pandas_tempfile1 = np.load('sample_mri_t.npy') # made up numpy
    #     pandas_tempfile2 = np.load('sample_mri_t.npy') # another made up numpy
    #     pandas_tempfile1 
    #     with TemporaryDirectory() as td:
    #         with open(os.path.join(td, tempfile1), 'w') as tf:
    #             tf.write('string')
    #         with open(os.path.join(td, tempfile2), 'w') as tf:
    #             tf.write('another_string')
            
    #         self.assertTrue(check_identical_columns(td).equals(0))
    pass
                            
class TestIntersectMethods(unittest.TestCase):

    def test_intersect_all(self):
        p = [[1,2,8,9], [1,2,3,4,5,6,7,8,9]]
        a = [[1,2,8,9], [1,2,3,4,5,6,7,8,9,10,11,12], [1,2,8,9,29]]
        intersect_p = intersect_all(*p)
        intersect_a = intersect_all(*a)
        self.assertEqual((intersect_p),(intersect_a))

class TestDockerTests(unittest.TestCase):

    def test_for_docker(self):
        p = np.load(sample_test_data1)
        a = np.load(sample_test_data2)
        self.assertEqual((p.sum() + 24),(a.sum()))


if __name__ == '__main__':
    unittest.main()