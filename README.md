<p align="center">
    <img style="width: 35%; height: 35%" src="cv_asl_logo.png">
</p>

[![DOI](To be made)
[![PyPI- to be made, placeholder](https://img.shields.io/pypi/v/cvasl.svg)](https://pypi.python.org/pypi/cvasl/)
[![Anaconda-Server Badge- to be made, placeholder](https://anaconda.org/brainspinner/cvasl/badges/version.svg)](https://anaconda.org/resurfemg/resurfemg)
[![Sanity](https://github.com/brainspinner/cvasl/actions/workflows/on-commit.yml/badge.svg)](https://github.com/brainspinner/cvasl/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/brainspinner/cvasl/actions/workflows/on-tag.yml/badge.svg)](https://github.com/brainspinner/cvasl/actions/workflows/on-tag.yml)

**cvasl** is an open source collaborative python library for analysis
of brain MRIs. Many functions relate to arterial spin labeled sequences.



This library
supports the ongoing research at University of Amsterdam Medical Center on brain ageing, but
is being buit for the entire community of radiology researchers across all university and academic medical centers and beyond.


### Program files

The main program in this repository (made of the modules in the cvasl folder) contains functions for analysis of MRIs..

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


### Configuring (to work with your data)

In order to preprocess and/or to train  models the code needs to be
able to locate the raw data you want it to find.

There are several ways to specify the location of the following
directories:

-   **root_mri_directory:** Special directory.  The rest of the directory layout can
    be derived from its location.
-   **preprocessed:** The directory that will be used by preprocessing
    code to output to.
-   **models:** The directory to output trained models to.

You can store this information persistently in several locations.

1.  In the same directory where you run the script (or the notebook).
    e.g. `./config.json`.
2.  In home directory, e.g. `~/.cvasl/config.json`.
3.  In global directory, e.g. `/etc/cvasl/config.json`.

However, we highly recommend you use the home directory.
This file can have this or similar contents:

    {
 
        'source_data': '/mnt/source_data',
        'raw_data': '/mnt/data/raw_data',
        'explore_asl': '/mnt/data/derivates/explore_asl',
        'cvage': '/mnt/data/derivates/cvage',
        'cvage_inputs': '/mnt/data/derivates/cvage/cvage_inputs',
        'cvage_outputs': '/mnt/data/derivates/cvage/cvage_outputs',

    }

The file is read as follows: if the file only specifies `raw_data`
directory, then the derivative missing entries are assumed to be relative to
the root in a BIDS compliant format order You don't need to specify all entries. If you do,
you can overwrite the ALS-BIDS format order but this is not reccomended.

### Test data

You can get test data by contacting the cvage team. Please email Dr. Moore at c.moore@esciencecenter.nl


## Getting started


How to get the notebooks running?  Assuming the raw data set and
metadata is available.

0. Assuming you are using conda for package management:    
  * Make sure you are in no environment:

      ```sh
      conda deactivate
      ```

      _(optional repeat if you are in the base environment)_

      You should be in no environment or the base environment now


1. Option A: To work with the most current versions with the possibility for development:
  Install all Python packages required, using `conda` and the `environment.yml` file. 


   * The command for Windows/Anaconda users can be something like:

     ```sh
     conda env create -f environment.yml
     ```

  Option B:
   * Linux users can create their own environment by hand (use
     install_dev as in setup).



### Testing

The project doesn't include testing data yet.  

### Command-Line Interface
You will eventually be be able to preprocess, train and use models, and perform other functions using command-line interface. As of now (April 2023) this module is still being built.

Below is an example of how to look at the help for that in general:
`python -m cvasl --help` 

And here is an example for a specific function:
`python -m cvasl hash_over --help`

And here are examples of a working commands (file names can be changed):
to hash over files:
`python -m cvasl hash_over --extension tsv  --input test_data --output some_ignored_folder`

to run a debiasing algorithm over files:
`python -m cvasl debias_over --preprocessing N4_debias_sitk --input test_data --output ignrd_flder `


All long options have short aliases.


✨Copyright 2023 Netherlands eScience Center and U. Amsterdam Medical Center
Licensed under <TBA> See LICENSE for details.✨
