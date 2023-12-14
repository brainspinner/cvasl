# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and
the Amsterdam University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for command line processing
of images and image derived data.
"""

import sys
import os
import logging
from argparse import ArgumentParser, ArgumentTypeError
from .file_handler import Config, hash_folder

from .mold import debias_folder
from .seperated import recode_sex_folder
from .seperated import make_log_folder
from .seperated import drop_columns_folder
from .seperated import make_log_file


def common(parser):
    """
    This function defines some arguments that can be called from any command
    line function to be defined.
    """
    parser.add_argument(
        '-o',
        '--output',
        default='.',
        help='''
        Directory containing algorithm output (created if doesn't exist).
        Defaults to current directory.
        ''',
    )


def config_override(override):
    if ':' not in override:
        raise ArgumentTypeError('Overrides must be of "key:value" format')
    key, path = override.split(':', 1)
    key = key.strip()
    path = os.path.realpath(path.strip())

    if key not in Config.default_layout:
        raise ArgumentTypeError(
            '{} is not valid, must be one of: {}'.format(
                key,
                tuple(Config.default_layout.keys()),
            ))
    return key, path


def make_parser():
    """
    This is the setting up parser for our CLI.
    """
    parser = ArgumentParser('CVASL')
    cfg_group = parser.add_mutually_exclusive_group()
    cfg_group.add_argument(
        '-c',
        '--config',
        default=None,
        help='''
        Location of config.json, a file that specified directory layout.
        This file is necessary to locate the data directory,
        models, preprocessed data and so forth.
        This option conflicts with -n.
        ''',
    )
    cfg_group.add_argument(
        '-n',
        '--no-config',
        action='store_true',
        default=False,
        help='''
        Specify this if no configuration loading is necessary.
        This implies that required keys in configuration will be provided by -C
        options.  This option conflict with -c.
        '''
    )
    parser.add_argument(
        '-C',
        '--config-override',
        default=[],
        action='append',
        metavar='KEY:PATH',
        type=config_override,
        help='''
        Override individual entry (path) in configuration file.  Repeatable.
        If used together with -n, this sets the value of the field rather
        than overriding it.
        Possible keys: {}
        '''.format(tuple(Config.default_layout.keys())),
    )
    subparsers = parser.add_subparsers()

    column_out_over = subparsers.add_parser('column_out_over')
    column_out_over.set_defaults(action='column_out_over')

    column_out_over.add_argument(
        '-i',
        '--input',
        default='.',
        help='''
        Folder to use csv data data.
        ''',
    )

    column_out_over.add_argument(
        '-r',
        '--removed_columns',
        action='append',
        default=['wmh_count'],
        help='''
        columns which will be removed.
        ''',
    )

    common(column_out_over)

    log_file_over = subparsers.add_parser('log_file_over')
    log_file_over.set_defaults(action='log_file_over')

    log_file_over.add_argument(
        '-i',
        '--input',
        default='.',
        help='''
        File to use csv data data.
        ''',
    )

    log_file_over.add_argument(
        '-l',
        '--columns',
        action='append',
        default=['wmh_count'],
        help='''
        columns which will be loged.
        ''',
    )

    common(log_file_over)

    log_recode_over = subparsers.add_parser('log_recode_over')
    log_recode_over.set_defaults(action='log_recode_over')

    log_recode_over.add_argument(
        '-i',
        '--input',
        default='.',
        help='''
        Folder to use csv data data.
        ''',
    )

    log_recode_over.add_argument(
        '-l',
        '--columns',
        action='append',
        default=['wmh_count'],
        help='''
        columns which will be loged.
        ''',
    )

    common(log_recode_over)

    sex_recode_over = subparsers.add_parser('sex_recode_over')
    sex_recode_over.set_defaults(action='sex_recode_over')

    sex_recode_over.add_argument(
        '-i',
        '--input',
        default='.',
        help='''
        Folder to use csv data data.
        ''',
    )

    common(sex_recode_over)

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

    dump_config = subparsers.add_parser('dump_config')
    dump_config.set_defaults(action='dump_config')

    return parser


def main(argv):
    """
    This runs the parser and subparsers.
    """
    parser = make_parser()
    parsed = parser.parse_args(argv)
    # import pdb
    # pdb.set_trace()

    if parsed.no_config:
        config = Config.no_file(parsed.config_override)
    else:
        config = Config.from_file(parsed.config, parsed.config_override)

    if parsed.action == 'sex_recode_over':
        try:
            recode_sex_folder(parsed.input)
        except Exception as e:
            logging.exception(e)
            return 1

    elif parsed.action == 'log_file_over':
        try:
            make_log_file(
                parsed.input,
                parsed.columns,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    elif parsed.action == 'log_recode_over':
        try:
            make_log_folder(
                parsed.input,
                parsed.columns,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    elif parsed.action == 'column_out_over':
        try:
            drop_columns_folder(
                parsed.input,
                parsed.removed_columns,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    elif parsed.action == 'hash_over':
        try:

            hash_folder(
                config.get_directory('raw_data'),
                # so the order here should switch?
                parsed.extension,
                parsed.output,
                parsed.force,
            )
        except Exception as e:
            logging.exception(e)
            return 1
    elif parsed.action == 'debias_over':
        try:

            debias_folder(
                config.get_directory('raw_data'),
                # so the order here should switch?
                parsed.preprocessing,
                parsed.output,
                # so the order here should switch?
                parsed.force,
            )
        except Exception as e:
            logging.exception(e)
            return 1

    elif parsed.action == 'dump_config':
        config.pprint(sys.stdout)

    # TODO(makeda): User didn't specify any action: what do we do?  Is
    # this legal?  Are there going to be more actions?
    return 0
