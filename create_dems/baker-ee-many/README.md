# Create DEMs using EE and NAGAP images

## Instructions

Run scripts in the following order, modifying inputs in the notebooks as necessary.

1. baker_ee_preprocess_datasets.py
2. baker_nagap_preprocess_datasets.py
3. baker_prepare_all_images.py
4. baker_run_timesift_pipeline.py

To Do:
Add description of process and scripts to use to:
1. run dem_align.py for all intermediate products
2. select the best products
3. create a dict containing paths/type of best products and run `timesift_pipeline.process_final_orthomosaics` as in 
    the process_best_dems_into_orthomosaics.py script
4. Create mosaic products (are we doing this? maybe not)
5. describe/include other scripts (which are run in this order):
        process_best_dems_into_orthomosaics.py
        collect_final_products.py
        prepare_dems_for_analysis.py

## Description of each script

**baker_ee_preprocess_datasets.py**
* Find image datasets available through Earth Explorer within a provided bounding box.
* Create a set of fidicual proxy template images for each set of images available.
* Prepare the images for SfM processing.

**baker_nagap_preprocess_datasets.py**
* Download and prep NAGAP images for SfM processing within a provided bounding box.

**baker_prepare_all_images.py**
* Move all the images into the same directory. This step is currently necessary because the EE preprocessing function is only set up to prep data to be run via hsfm.batch.batch_process, not to be run with hsfm.pipeline.TimesiftPipeline or hsfm.pipeline.Pipeline.

**baker_run_timesift_pipeline.py**
* Set up and run a timesift pipeline.

