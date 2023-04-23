"""
Copyright 2023 Netherlands eScience Center and University of Amsterdam.
Licensed under <TBA>. See LICENSE for details.

This file contains one method to let the user configure
all paths for data instead of hard-coding them, as well
as methods to check data integrity, and create synthetic data.
The data integrity can be checked because this file contains
hash functions to track data. Synthetic data can be made
with several methods.
"""
# import libraries


import json
import logging
import os
import textwrap
import hashlib
import glob
import math
import pandas as pd
import numpy as np
import scipy
from scipy import signal
import random


class Config:
    """
    This class allows configuration on the home computer
    or remote workspace, of a file setup for data,
    which is then processed into a variable. Essentially
    by setting up and modifying a .json file in the appropriate directory
    users can avoid the need for any hardcoded paths to data.
    If you do not set up a json file, then
    your storage space will default to the test_data folder
    in the repository.
    """

    default_locations = (
        './config.json',
        os.path.expanduser('~/.cvasl/config.json'),
        '/etc/cvasl/config.json',
    )

    default_layout = {
        'bids': '{}',
        'raw_data': '{}/raw_data',
        'derivatives': '{}/derivatives',
        'explore_asl': '{}/derivatives/explore_asl',
        'cvage': '{}/derivates/cvage',
        'cvage_inputs': '{}/derivates/cvage/cvasl_inputs',
        'cvage_outputs': '{}/derivates/cvage/cvasl_outputs',
    }

    required_directories = 'bids', 'raw_data', 'derivatives'

    def __init__(self, location=None):
        self._raw = None
        self._loaded = None
        self.load(location)
        self.validate()

    def usage(self):
        """
        This is essentally anotice message if the computer
        does not have paths configured or files made so that
        the data paths of a config.json can be used.
        Until you do it will defailt to test_data
        """
        return textwrap.dedent(
            '''
            Cannot load config. If you did not make a config,
            until you do your data layout not be accessed.
            If you tried to make a config.json it is not in the
            right place.

            Please create a file in either one of the locations
            listed below:
            {}

            With the contents that specifies at least the root
            directory, and other neccesary directories as follows:

            {{
                "bids": "/path/to/storage"
                "raw_data": "/path/to/storage/raw_data",
                "derivatives": "/path/to/storage/derivatives/",
            }}

            The default directory layout is expected to be based on the above
            and adding subdirectories.

            You can override any individual directory (or subdirectory)
            by specifying it in the config.json file.

            {} are expected to exist.
            '''
        ).format(
            '\n'.join(self.default_locations),
            self.required_directories,
        )

    def load(self, location):
        locations = (
            [location] if location is not None else self.default_locations
        )

        found = None
        for p in locations:
            try:
                with open(p) as f:
                    self._raw = json.load(f)
                    found = p
                    break
            except json.JSONDecodeError as e:
                raise ValueError(
                    'Cannot parse configuration in {}'.format(p),
                ) from e
            except Exception as e:
                logging.info('Configuration not found in %s: %s', p, e)
        else:
            raise ValueError(self.usage())

        root = self._raw.get('bids')
        self._loaded = dict(self._raw)
        if root is None:
            required = dict(self.default_layout)
            del required['bids']
            for directory in required.keys():
                if directory not in self._raw:
                    raise ValueError(
                        'Configuration in {} is missing required directory {}'
                        .format(found, directory),
                    )
            # User specified all concrete directories.  Nothing for us to
            # do here.
        else:
            missing = set(self.default_layout.keys()) - set(self._raw.keys())
            # User possibly specified only a subset of directories.  We'll
            # back-fill all the not-specified directories.
            for m in missing:
                self._loaded[m] = self.default_layout[m].format(root)

    def validate(self):
        # These directories are required to exist (contrast with the
        # loading code where we check for user *specifying* required
        # directories)
        for d in self.required_directories:
            if not os.path.isdir(self._loaded[d]):
                raise ValueError(
                    'Required directory {}: {} doesn\'t exist'.format(
                        d,
                        self._loaded[d],
                    ))

    def get_directory(self, directory, value=None):
        if value is None:
            return self._loaded[directory]
        return value


def hash_folder(origin_folder1, file_extension, made, force=False):
    """Hashing function to be used by command line.

    :param origin_folder1: The string of the folder with files to hash
    :type origin_folder1: str
    :param file_extension: File extension
    :type file_extension: str
    :param made: file directory where csv with hashes will be put
    :type made: str
    """
    filepath = os.path.join(made, 'hash_output.csv')
    df = hash_rash(origin_folder1, file_extension)
    if not force:
        if os.path.isfile(filepath):
            return
    try:
        os.makedirs(os.path.dirname(filepath))
    except FileExistsError:
        pass

    df.to_csv(filepath)

# def save_preprocessed(array, out_fname, force):
#     """
#     This function is written to be called by the cli module.
#     It stores arrays in a directory.
#     """
#     if not force:
#         if os.path.isfile(out_fname):
#             return
#     try:
#         os.makedirs(os.path.dirname(out_fname))
    # except FileExistsError:
    #     pass
    # np.save(out_fname, array, allow_pickle=False)


def hash_rash(origin_folder1, file_extension):
    """Hashing function to check files are not corrupted or to assure
    files are changed.

    :param origin_folder1: The string of the folder with files to hash
    :type origin_folder1: str
    :param file_extension: File extension
    :type file_extension: str

    :returns: Dataframe with hashes for what is in folder
    :rtype: ~pandas.DataFrame
    """
    hash_list = []
    file_names = []
    files = '**/*.' + file_extension

    non_suspects1 = glob.glob(
        os.path.join(origin_folder1, files),
        recursive=True,
    )
    BUF_SIZE = 65536
    for file in non_suspects1:
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        hash_list.append(result)
        file_names.append(file)

    df = pd.DataFrame(hash_list, file_names)
    df.columns = ["hash"]
    df = df.reset_index()
    df = df.rename(columns={'index': 'file_name'})

    return df