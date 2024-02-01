# Change Log

## Unreleased

Changed repo name from brainspin to cvasl
### Added
* cli module for command line interface
* file_handler module for configuring and hashing data
* carve module for handeling DICOM data
* mold module for plotting and debiasing images
* seperated module for dealing csv and tsv derived data
* a changelog
* tests/test.py for testing

### Changed

*
## Release 0.0.0-alpha

### Added

* First version of this Python project to follow the Netherlands eScience Center software development guide, containing (added to this version):

	- License
    - Notice
    - seperate environments for harmonization experiments inside harmonization_paper folder
    - notebook showing how showable standard was generated
    - file with standard for submitting to package inside researcher_interface folder
    - preliminary linear regression models as baseline for brain age models

## Release 0.0.1

### Added

* Second version of this Python project to follow the Netherlands eScience Center software development guide, containing (added to this version):

    - additional experiments inside harmonization_paper folder
    - example of correct format for csv of derived values to use with library 

## Release 0.0.2

### Added

* Third version of this Python project containing (added to this version):

	- haromnzation_abstract_one folder to save and make reproducible work for brain mapping abstract 
    - exact freeze of environment in precise_working_environment

    - harmony module for functions related to implementing common harmonization algorithms
    - harmony also includes new graphing functions in harmony module to illustrate harmonization effects e.g.
    `compare_harm_one_site_violins` and `compare_harm_multi_site_violins`

    - updates to seperated module including:
    - new functons in seperated to bin data on a continous variable
    - generalized functions for k-folding as specific to project in seperated module
    - k folding which allows splitting on two variables (one continuous, one categorical) with           
    `stratified_cat_and_cont_categories_shuffle_split` function 

    - command-line sex recoding over a folder
    - command-line loging of columns over a folder
    - command-line loging of columns over a file
    - command-line cleaning off unwanted columns run over a folder


## Release 0.0.3

### Pending

* Fourth version of this Python project containing (added to this version):

	- testing has been changed to use pytest ( all unit-test formatted tests will still run)
    - testing has been split into two files so CI can run a docker based testing, and a general (multi-os) test
    - possibility to test notebooks with nbmake module added, but not in CI due to time and data issues
    - vendor module of cvasl, harmonization code from outside packages without a release

