"""
Copyright 2023 Netherlands eScience Center and University of Amsterdam.
Licensed under <TBA>. See LICENSE for details.

This file contains methods to anonymize image files.
Specifically we have a methods to get the image out of DICOMs,
and to strip soft tissues off head MRIs.
"""
# import libraries


import os 
import glob
import numpy 
