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

### Pending (December 11 2023)

* Third version of this Python project to follow the Netherlands eScience Center software development guide, containing (added to this version):

	- harmony module for functions related to common harmonization algorithms
    - generalized function for k-folding as specific to project in seperated module
    - exact freeze of environment in precise_working_environment
    - new graphing functions in harmony module to illustrate harmonization effects
    - (pending) k folding which allows splitting on a continous variable?