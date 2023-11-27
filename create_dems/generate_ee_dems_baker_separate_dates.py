#%%
import pandas as pd
from hsfm import pipeline

## Split image csv by dates


## Run for 09/02 bunch of images
pipeline1 = pipeline.Pipeline(
    input_images_path = "/data2/elilouis/timesift/baker-ee-backup/preprocessed_images",
    reference_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/copernicus_joined_dem.tif",
    image_matching_accuracy = 4,
    densecloud_quality = 4,
    output_DEM_resolution = 30,
    project_name = "eeproject_lk000_09_02_1950",
    output_path = "/data2/elilouis/timesift/baker-ee-backup/mixed_timesift/individual_clouds/50_9.0/manual_cluster_09_02/",
    input_images_metadata_file = "/data2/elilouis/timesift/baker-ee-backup/mixed_timesift/individual_clouds/50_9.0/manual_cluster_09_02/metashape_metadata.csv",
    camera_models_path="/data2/elilouis/timesift/baker-ee-backup/mixed_timesift/multi_epoch_cloud/camera_calibrations",
    license_path="uw_agisoft.lic",
    verbose=False,
    rotation_enabled=True,
)

s## Run for 08/17 bunch of images
#%%
pipeline2 = pipeline.Pipeline(
    input_images_path = "/data2/elilouis/timesift/baker-ee-backup/preprocessed_images",
    reference_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/copernicus_joined_dem.tif",
    image_matching_accuracy = 4,
    densecloud_quality = 4,
    output_DEM_resolution = 30,
    project_name = "eeproject_lk000_08_17_1950",
    output_path = "/data2/elilouis/timesift/baker-ee-backup/mixed_timesift/individual_clouds/50_9.0/manual_cluster_08_17/",
    input_images_metadata_file = "/data2/elilouis/timesift/baker-ee-backup/mixed_timesift/individual_clouds/50_9.0/manual_cluster_08_17/metashape_metadata.csv",
    camera_models_path="/data2/elilouis/timesift/baker-ee-backup/mixed_timesift/multi_epoch_cloud/camera_calibrations",
    license_path="uw_agisoft.lic",
    verbose=False,
    rotation_enabled=True,
)

pipeline1.run()
pipeline2.run()