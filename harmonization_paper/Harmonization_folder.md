## Harmonization paper computation

This folder contains notebooks and environments related to work for a paper on image harmonization. Several libraries used e.g. neurocombat require special environments that differ from our regular environments. They can be built as follows (here the neurocombat environment is used as an example):

0. Assuming you are using conda for package management:    
  * Make sure you are in no environment:

      ```sh
      conda deactivate
      ```

      _(optional repeat if you are in the base environment)_

      You should be in no environment or the base environment now


1. 
  Option A: Fastest option:
  In a base-like environment with mamba installed, you can install all Python packages required, using `mamba` and the `environment.yml` file. 

  If you do not have mamba installed you can follow instructions (here)[https://anaconda.org/conda-forge/mamba]
  


   * The command for Windows/Anaconda/Mamba users can be something like:

     ```sh
     mamba env create -f harmonization_paper/neurocombat_environment.yml
     ```

    Option B: Do it with only conda
