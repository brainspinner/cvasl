"""
Copyright 2023 Netherlands eScience Center and
the Amsterdam University Medical Center.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to normalize brain MRIs.
"""
# import libraries


import os
import glob
import numpy as np
import importlib
import matplotlib.pyplot as plt


class NormedSliceViewer:
    """
    A class to examine slices of MRIs, or other volumetric data,
    with normalizaiotn

    """
    # TODO: should just be an option with widget in slice viewer
    def __init__(self, volume, figsize=(10, 10)):
        self.volume = volume
        self.figsize = figsize
        self.v = [np.min(volume), np.max(volume)]
        self.widgets = importlib.import_module('ipywidgets')

        self.widgets.interact(self.transpose, view=self.widgets.Dropdown(
            options=['axial', 'sag', 'cor'],
            value='axial',
            description='View:',
            disabled=False))

    def transpose(self, view):
        # transpose the image to orient according to the slice plane selection
        orient = {"sag": [1, 2, 0], "cor": [2, 0, 1], "axial": [0, 1, 2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1

        self.widgets.interact(
            self.plot_slice,
            z=self.widgets.IntSlider(
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


class SliceViewer:
    """
    A class to examine slices of MRIs, or other volumetric data

    """
    def __init__(self, volume, figsize=(10, 10)):
        self.volume = volume
        self.figsize = figsize
        self.v = [np.min(volume), np.max(volume)]
        self.widgets = importlib.import_module('ipywidgets')

        self.widgets.interact(self.transpose, view=self.widgets.Dropdown(
            options=['axial', 'sag', 'cor'],
            value='axial',
            description='View:',
            disabled=False))

    def transpose(self, view):
        # transpose the image to orient according to the slice plane selection
        orient = {"sag": [1, 2, 0], "cor": [2, 0, 1], "axial": [0, 1, 2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1

        self.widgets.interact(
            self.plot_slice,
            z=self.widgets.IntSlider(
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
            vmin=0,
            vmax=255,
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
                      "8-bit signed integer": sitk.sitkInt8,
                      "Signed 16 bit integer": sitk.sitkInt16,
                      "16-bit signed integer": sitk.sitkInt16,
                      "Signed 32 bit integer": sitk.sitkInt32,
                      "32-bit signed integer": sitk.sitkInt32,
                      "Signed 64 bit integer": sitk.sitkInt64,
                      "64-bit signed integer": sitk.sitkInt64,
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


def save_preprocessed(array, out_fname, force):
    """
    This function is written to be called by the cli module.
    It stores arrays in a directory.
    """
    if not force:
        if os.path.isfile(out_fname):
            return
    try:
        os.makedirs(os.path.dirname(out_fname))
    except FileExistsError:
        pass
    np.save(out_fname, array, allow_pickle=False)


def debias_folder(file_directory, algorithm, processed, force=False):

    """
    Debias  function to perform bias field correction over an entire folder,
    through command_line. It does not return, files made are an artifact.
    Note this will only run on files with .gz at end of extension.

    :param file_directory: The string of the folder with files to hash
    :type file_directory: str
    :param algorithm: algorithm e.g. N4_debias_sitk
    :type algorithm: algorithm
    :param processed: folder where output images go
    :type processed: str

    """
    file_directory_list = glob.glob(
        os.path.join(file_directory, '**/*.gz'),
        recursive=True,
    )
    for file in file_directory_list:
        if algorithm == 'n4_debias_sitk':
            array = n4_debias_sitk(file)
        elif algorithm == 'alternative_debias_a':
            array = alternative_debias_a(file)

        else:
            array = n4_debias_sitk(file)
        rel_fname = os.path.relpath(file, file_directory)
        out_fname = os.path.join(processed, rel_fname)
        save_preprocessed(array, out_fname, force)
