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
#hash_rash
from cvasl.file_handler import hash_rash
from cvasl.file_handler import intersect_all


sample_test_data1 = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    '.helpers/fake_test_data',
    'ar1.npy',
)

sample_test_data2 = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(__file__))),
    '.helpers/fake_test_data',
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