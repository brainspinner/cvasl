# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and VUMC(?).
Licensed under <TBA>. See LICENSE for details.

This file contains functions for command line processing
"""

import logging
from argparse import ArgumentParser
from .file_handler import Config

from .file_handler import hash_folder
from .mold import debias_folder


def common(parser):
    """
    This function defines some arguments that can be called from any command
    line function to be defined.
    """
    parser.add_argument(
        '-i',
        '--input',
        default=None,
        help='''
        Directory containing files to be worked on
        ''',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='''
        Directory containing algorithm output (created if doesn't exist).
        ''',
    )


def make_parser():
    """
    This is the setting up parser for our CLI.
    """
    parser = ArgumentParser('CLI')
    parser.add_argument(
        '-c',
        '--config',
        default=None,
        help='''
        Location of config.json, a file that specified directory layout.
        This file is necessary to locate the data directory,
        models, preprocessed data and so forth.
        '''
    )
    subparsers = parser.add_subparsers()
    hash_over = subparsers.add_parser('hash_over')
    hash_over.set_defaults(action='hash_over')

    hash_over.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='''
        Write over previously hashed data.
        ''',
    )

    hash_over.add_argument(
        '-x',
        '--extension',
        action='store',
        default=[],
        help='''
        Extension of files to be hashed.
        ''',
    )

    common(hash_over)

    debias_over = subparsers.add_parser('debias_over')
    debias_over.set_defaults(action='debias_over')

    debias_over.add_argument(
        '-f',
        '--force',
        action='store_true',
        default=False,
        help='''
        Write over previously preprpocessed data.
        ''',
    )

    debias_over.add_argument(
        '-p',
        '--preprocessing',
        default='N4_debias_sitk',
        choices=(
            'N4_debias_sitk',
            'alternative_debias',
            'something_else_allowed'),
        type=str,
        help='''
        Pick the desired algorithm for bias field preprocessing.
        ''',
    )
    common(debias_over)

    return parser


def main(argv):
    """
    This runs the parser and subparsers.
    """
    parser = make_parser()
    parsed = parser.parse_args()
    config = Config(parsed.config)  # can be
    # rewritten to take input or config file, but not really neccesary
    if parsed.action == 'hash_over':
        try:

            hash_folder(
                config.get_directory('data', parsed.input),
                # so the order here should switch?
                parsed.extension,
                parsed.output,
                parsed.force,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    if parsed.action == 'debias_over':
        try:

            debias_folder(
                config.get_directory('data', parsed.input),
                # so the order here should switch?
                parsed.preprocessing,
                config.get_directory('preprocessed', parsed.output),
                # so the order here should switch?
                parsed.force,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    return 0
