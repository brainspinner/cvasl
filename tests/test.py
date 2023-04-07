#sanity tests for the  library


import unittest
import os
import numpy as np
import scipy
from tempfile import TemporaryDirectory
import json
from unittest import TestCase, main

# config
from cv_asl.file_handler import Config

class TestConfig(TestCase):

    required_directories = {
        'root_mri_directory',
    }
    required_directories = ['root_mri_directory']

    def test_roots_only(self):
        with TemporaryDirectory() as td:
            same_created_path = os.path.join(td, 'root')
            os.mkdir(same_created_path)
            raw_config = {
                'root_mri_directory': same_created_path,
            }
            config_file = os.path.join(td, 'config.json')
            with open(config_file, 'w') as f:
                json.dump(raw_config, f)

            # for root in self.required_directories:
            #     os.mkdir(os.path.join(td, root))

            config = Config(config_file)
            assert config.get_directory('root_mri_directory')

    def test_missing_config_path(self):
        try:
            Config('non existent')
        except ValueError:
            pass
        else:
            assert False, 'Didn\'t notify on missing config file'
    

        


if __name__ == '__main__':
    unittest.main()