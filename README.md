<p align="center">
    <img style="width: 35%; height: 35%" src="">
    🧠
</p>

[![DOI](To be made)
[![PyPI- to be made, placeholder](https://img.shields.io/pypi/v/brainspin.svg)](https://pypi.python.org/pypi/brainspin/)
[![Anaconda-Server Badge- to be made, placeholder](https://anaconda.org/brainspinner/brainspin/badges/version.svg)](https://anaconda.org/resurfemg/resurfemg)
[![Sanity](https://github.com/brainspinner/brainspin/actions/workflows/on-commit.yml/badge.svg)](https://github.com/brainspinner/brainspin/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/brainspinner/brainspin/actions/workflows/on-tag.yml/badge.svg)](https://github.com/brainspinner/brainspin/actions/workflows/on-tag.yml)

**brainspin** is an open source collaborative python library for analysis
of brain MRIs. Many functions relate to arterial spin labeled sequences.



This library
supports the ongoing research at University of Amsterdam Medical Center on brain ageing, but
is being buit for the entire community of radiology researchers across all university and academic medical centers and beyond.



## Getting started

### Installation

How to get the notebooks running?  Assuming the raw data set and
metadata is available.

0. Assuming you are using conda for package management:    
  * Make sure you are in no environment:

      ```sh
      conda deactivate
      ```

      _(repeat if you are in the base environment)_

      You should be in no environment now


1. Option A: To work with the most current versions with the possibility for development:
  Install all Python packages required, using `conda` and the `environment.yml` file. 


   * The command for Windows/Anaconda users can be something like:

     ```sh
     conda env create -f environment.yml
     ```

  Option B:
   * Linux users can create their own environment by hand (use
     install_dev as in setup).


### Configuring (to work with your data)

In order to preprocess and/or to train  models the code needs to be
able to locate the raw data you want it to find.

There are several ways to specify the location of the following
directories:

-   _root_mri_directory_: Special directory.  The rest of the directory layout can
    be derived from its location.
-   _preprocessed_: The directory that will be used by preprocessing
    code to output to.
-   _models_: The directory to output trained models to.


The root directory of this repository contains a file called `config.json`, which by default stores the `/test_data` folder of this repository as your _root_mri_directory_. You can copy your (test) data to this location or adjust the _root_mri_directory_ to another location where your data is stored. 
Notes:
- while in principle no files in the `/test_data` folder will be pushed to your online repository, we recommend not storing large amounts or sensitive data within your repository folder structure.  
- if you do not like to store your _config.json_ file within the repository (as this displays your system's folder structure), you can choose to store your config.json outside of the repository, in your home directory:
  - on Windows: `C:\Users\<YourUserName>\.brainspin\config.json`
  - on Linux: `home/.brainspin/config.json`
  - on Mac: ???


{
    "root_mri_directory": "/mnt/data",
    "preprocessed": "/mnt/data/preprocessed",
    "models": "/mnt/data/models",
    "output": "/mnt/data/output",
}

The file is read as follows: if the files specifies `root_mri_directory`
directory, then the missing entries are assumed to be relative to
the root.  You don't need to specify all entries.


### Test data

You can get test data by ... (TBA)



## Repository organization

### Program files

The main program in this repository (made of the modules in the brainspin folder) contains functions for analysis of MRIs..

### Folders and Notebooks

To look around keep in mind the following distinction on folders:

researcher_interface:
- These are a growing series of interactive notebooks that allow
  researchers to investigate questions about their own MRI data
 
open_work:
- This folder contains experimental work by core members of the brainage
  team (Dr. Candace Makeda Moore, Dr. Dani Bodor, Dr. Henk Mutsaerts)


### Data sets

The notebooks are configured to run on various datasets.  Contact
Dr. Candace Makeda Moore( 📫 c.moore@esciencecenter.nl) to discuss any
questions on data configuration for your datasets.



### Testing

The project doesn't include testing data yet.  

✨Copyright 2023 Netherlands eScience Center and U. Amsterdam Medical Center
Licensed under <TBA> See LICENSE for details.✨
