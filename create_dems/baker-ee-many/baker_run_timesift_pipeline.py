# Mt Baker
# Run timesift pipeline
# 
# This notebook prepares a TimesiftPipeline object for all the EE and NAGAP images downloaded in previous notebooks.
# The TimesiftPipeline is then run step by step to allow for specification of DEM resolution for different image datasets.

import os
from hsfm.pipeline import TimesiftPipeline

# Required inputs:
main_directory = '/data2/elilouis/timesift/baker-ee-many/'
combined_metadata_file_path = '/data2/elilouis/timesift/baker-ee-many/combined_metashape_metadata.csv'
combined_image_metadata_file_path = '/data2/elilouis/timesift/baker-ee-many/combined_image_metadata.csv'
preprocessed_images_dir = '/data2/elilouis/timesift/baker-ee-many/preprocessed_images'
timesift_output_path = '/data2/elilouis/timesift/baker-ee-many/mixed_timesift'
reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_10m.tif'
reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif'
license_path = "/home/elilouis/hsfm/uw_agisoft.lic"

# Create pipeline input
timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file = combined_metadata_file_path,
    image_metadata_file = combined_image_metadata_file_path,
    raw_images_directory = preprocessed_images_dir,
    output_directory = timesift_output_path,
    reference_dem_lowres = reference_dem_lowres,
    reference_dem_hires = reference_dem_hires,
    image_matching_accuracy = 1,
    densecloud_quality = 2,
    output_DEM_resolution = 5,
    license_path=license_path,
    parallelization=1
)

# The entire pipeline can be run with one command (see line below), but this is not preferable 
# if you have images that allow for DEMs of different final resolutions to be generated.
# Because we expect this, we manually run each step that is called within TimesiftPipeline.run()
# and, when processing individual dates, we specify the DEM resolution that we would like for each
# dataset. We determined this by looking at Metashape reports that are generated during the clustering 
# step. Specifically in the report we look at the ground sample distance.

# timesift_pipeline.run()

# Run pipeline step by step so you can choose different output resolutions for the different dates

# metadata_timesift_aligned_file, unaligned_cameras_df = timesift_pipeline._generate_multi_epoch_densecloud()

# _ = timesift_pipeline._save_image_footprints()

# _ = timesift_pipeline._export_camera_calibration_files()

# _ = timesift_pipeline._prepare_single_date_data(metadata_timesift_aligned_file)

# dict_of_subsets_by_date = timesift_pipeline._find_clusters_in_individual_clouds()

# print('Image subsets organized by date:')
# print(dict_of_subsets_by_date)

# _ = timesift_pipeline._generate_subsets_for_each_date(dict_of_subsets_by_date)

# Run everything above this point first. Then look at all the Metashape reports and decide what DEM
# resolutions we want for each individual dataset. Modify the values for output_DEM_resolution_override 
# as shown below.

_ = timesift_pipeline._process_individual_clouds(
    ['47_9.0_14.0', '50_9.0_2.0'],
     output_DEM_resolution_override=4
)

_ = timesift_pipeline._process_individual_clouds(
    ['72_8.0_10.0'], output_DEM_resolution_override=10
)

_ = timesift_pipeline._process_individual_clouds(
    ['79_9.0_14.0'], output_DEM_resolution_override=7
)

_ = timesift_pipeline._process_individual_clouds(
    [
        '67_9.0_21.0',
        '70_9.0_9.0',
        '70_9.0_29.0',
        '74_8.0_10.0',
        '75_9.0_7.0',
        '77_9.0_27.0',
        '79_10.0_6.0',
        '87_8.0_21.0',
        '88_8.0_21.0',
        '90_9.0_5.0',
        '91_9.0_9.0',
        '92_9.0_15.0',
        '92_9.0_18.0'
    ], 
    output_DEM_resolution_override=1
)

results_report_file = timesift_pipeline.create_results_report()
print(f'Results report at: {results_report_file}')
timesift_pipeline.create_mosaics(10)