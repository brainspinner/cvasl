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



class SliceViewer:
    """ 
    A class to examine slices of MRIs, or other volumetric data

    """
    def __init__(self, volume, figsize=(10,10)):
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
        orient = {"sag":[1,2,0], "cor":[2,0,1], "axial": [0,1,2]}
        self.vol = np.transpose(self.volume, orient[view])
        maxZ = self.vol.shape[2] - 1
        
        widgets.interact(self.plot_slice, # slider
            z=widgets.IntSlider(min=0, max=maxZ, step=1, continuous_update=True, 
            description='Image Slice:'))
        
    def plot_slice(self, z):
        # plot slice for plane which will match the widget intput
        self.fig = plt.figure(figsize=self.figsize)
        plt.imshow(self.vol[:,:,z], cmap="gray", 
            vmin=self.v[0], vmax=self.v[1])




