
# -*- coding: utf-8 -*-

"""
Copyright 2023 Netherlands eScience Center and VUMC(?).
Licensed under <TBA>. See LICENSE for details.

This file contains functions for processing csv and tsv
files towards correct formats.
"""

import pandas as pd # 
import numpy as np    
import copy     # Can Copy and Deepcopy files so original file is untouched.


def recode_sex(whole_dataframe, string_for_sex):
    """
    This function recodes sex into a new column if there 
    are two possible values. It maintains numerical older
    butchanges the values to 0 and 1. The new column is
    called 'sex_encoded'.

    :param whole_dataframe: dataframe variable
    :type whole_dataframe: str
    :param string_for_sex: column name written in singe qoutes
    :type string_for_sex: str

    :returns: dataframe with sex encoded colum
    :rtype: pandas.dataFrame
    
    """
    new_dataframe = whole_dataframe.copy()
    dataframe_column = whole_dataframe[string_for_sex]
    recoded = dataframe_column.copy()
    if len(dataframe_column.unique()) == 2:
        if recoded.unique()[0] < recoded.unique()[1]:
            # transform smaller number to zero
            recoded = recoded.replace(recoded.unique()[0], 0 )
            recoded = recoded.replace(recoded.unique()[1], 1 )
        else:
            recoded = recoded.replace(recoded.unique()[1], 0 )
            recoded = recoded.replace(recoded.unique()[0], 1 )
        new_dataframe['sex_encoded'] = recoded
    elif len(recoded.unique) < 2:
        print('there are two sexes, you have fewer, caution, encode by hand')
    else:
        print('there are two sexes, you have more, caution, encode by hand')
        
    return new_dataframe