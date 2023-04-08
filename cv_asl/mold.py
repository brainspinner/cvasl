"""
Copyright 2023 Netherlands eScience Center and University of Amsterdam.
Licensed under <TBA>. See LICENSE for details.

This file contains methods to normalize brain MRIs.
"""
# import libraries


import os
import glob
import numpy as np
from ipywidgets import IntSlider, Output
import ipywidgets as widgets
import matplotlib.pyplot as plt

import SimpleITK as sitk


class SliceViewer:
    """
    A class to examine slices of MRIs, or other volumetric data

    """
    def __init__(self, volume, figsize=(10, 10)):
        self.volume = volume
        self.figsize = figsize
        self.v = [np.min(volume), np.max(volume)]

        widgets.interact(self.transpose, view=widgets.Dropdown(
            options=['axial', 'sag', 'cor'],
            value='axial',
            description='View:',
            disabled=False))

    def transpose(self, view):
        # transpose the image to orient according to the slice plane selection
        orient = {"sag": [1, 2, 0], "cor": [2, 0, 1], "axial": [0, 1, 2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1

        widgets.interact(
            self.plot_slice,
            z=widgets.IntSlider(
                min=0,
                max=maxZ,
                step=1,
                continuous_update=True,
                description='Image Slice:'
            )
        )

    def plot_slice(self, z):
        # plot slice for plane which will match the widget intput
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(
            self.vol[:, :, z],
            cmap="gray",
            vmin=self.v[0],
            vmax=self.v[1],
        )


def n4_debias_sitk(
        image_filename,
        iteration_vector=[20, 10, 10, 5],
        masking=True
):
    """
    This is our implementation of sitk's N4 debiasing algorithm.
    It is implemeted so the algorithm
    can be applied unformly from command line (eventually)
    Need to cite SITK
    """
    # TODO: add sitk citation in docstring,
    inputImage = sitk.ReadImage(image_filename)
    bits_in_input = inputImage.GetPixelIDTypeAsString()
    bit_dictionary = {"Signed 8 bit integer": sitk.sitkInt8,
                      "Signed 16 bit integer": sitk.sitkInt16,
                      "Signed 32 bit integer": sitk.sitkInt32,
                      "Signed 64 bit integer": sitk.sitkInt64,
                      "Unsigned 8 bit integer": sitk.sitkUInt8,
                      "Unsigned 16 bit integer": sitk.sitkUInt16,
                      "Unsigned 32 bit integer": sitk.sitkUInt32,
                      "Unsigned 64 bit integer": sitk.sitkUInt64,
                      "32-bit float": sitk.sitkFloat32,
                      "64-bit float": sitk.sitkFloat64, }
    bits_ing = bit_dictionary[bits_in_input]
    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

    inputImage = sitk.Cast(inputImage, bits_ing)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()

    corrector.SetMaximumNumberOfIterations(iteration_vector)

    if masking:
        output = corrector.Execute(inputImage, maskImage)
    else:
        output = corrector.Execute(inputImage)
    outputCasted = sitk.Cast(output, bits_ing)

    return outputCasted
