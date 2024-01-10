# hsfm-geomorph
Using HSFM (historical structure from motion) DEMs to understand decadal-scale geomorphology in the PNW.

This library is composed of a number of scripts that you can run to analyze DEMs and produce plots, as published in the paper

https://doi.org/10.1016/j.geomorph.2023.108805

## Prerequisites
To run the scripts, you need:
1. A dataset, which is published (and downloadable) here: https://zenodo.org/records/10472311
2. A forked and modified version of the `xdem` library, which can be cloned from here: https://github.com/elischwat/xdem

## Instructions
Once you have the dataset downloaded and the forked version of xdem downloaded, follow these steps:

1. Install the conda environment, `conda env create -f environment.yml` or `mamba env create -f environment.yml`
2. Install the local version of xdem, `conda activate hsfm; pip install ~/xdem/` (you may need to modify the path to xdem for your system)
3. Export a variable containing the path to the downloaded dataset (decompressed). For example:
    ```
    $ export HSFM_GEOMORPH_DATA_PATH='/storage/username/'
    ```
    The downloaded dataset should be placed at this path, e.g. `/storage/username/hsfm-geomorph`

4. Navigate to the working directory and run the python scripts using the `run_all.sh` script:
    ```
    $ cd dem-analysis/mt_baker_mass_wasted/
    $ chmod +x run_all.sh
    $ ./run_all.sh
    ```

5. To reproduce the published figures, you can open the python scripts as jupyter notebooks and run them. To convert .py scripts to jupyter notebooks, see the documentation for jupytext, https://jupytext.readthedocs.io, which should already be installed as it's in the environment.yml.

## Notes
To closely reproduce the published numbers, you will need to make some changes to the python scripts and input files. Some parameter values were modified in this version of the code so that when the scripts are run, computational resources are not overwhelmed. To run the scripts with the modifications described below, we recommend using a machine with at least 64Gb RAM and with enough cores to run 64 parallel threads.

1. Change the following variables in all of the input files (files in dem-analysis/mt_baker_mass_wasted/inputs/). Note that these variables were modified to accomodate computer system limitations. This large subsample allows accurate estimation of spatial autocorrelation. For more information on this, see the documentation for xdem (https://xdem.readthedocs.io/en/stable/), we utilize the function `xdem.spatialstats.sample_empirical_variogram`. 
   * Set ["uncertainty"]["VARIOGRAM_SUBSAMPLE"] to 10000 (currently 100)
   * Set ["uncertainty"]["PARALLELISM"] to 64 (currently 10)

2. Change the value of the variables `VARIOGRAM_SUBSAMPLE` and `PARALLELISM` in the script `uncertainty_whole_mountain_combined.py` to match the values described above, 10000 and 64, respectively.

3. Change the value of the variable `RESAMPLING_RES` to 2 (from the current value of 10).