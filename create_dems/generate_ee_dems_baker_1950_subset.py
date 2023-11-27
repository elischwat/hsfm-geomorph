# # Generate EE DEMs, Baker

# +
# import shapely
# from shapely import wkt
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import math
# from itertools import chain

import os
import hipp
import pandas as pd
from getpass import getpass
from hsfm.pipeline import NAGAPPreprocessingPipeline
# -

license_path = "/home/elilouis/hsfm/uw_agisoft.lic"
output_directory = '/data2/elilouis/timesift/baker-ee'
nagap_image_info_fn = '/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata_updated_manual.csv'

# mkdir {output_directory}

# Run NAGAP preprocessing 
preprocess_pipeline = NAGAPPreprocessingPipeline(
    output_directory,
    fiducial_templates_directory = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/",
    nagap_images_csv_path = nagap_image_info_fn,
    bounds = (-121.94, 48.84, -121.70, 48.70)
)
preprocess_pipeline.run()

images_directory = os.path.join(output_directory, "raw_images")
ee_preprocessed_images_path = os.path.join(output_directory, "preprocessed_images_ee")
nagap_preprocessed_images_path = os.path.join(output_directory, "preprocessed_images")
nagap_metadata_file = os.path.join(output_directory, "metashape_metadata.csv")

# # Run the find_baker_ee_image_dataset.py script to generate this csv file.
# # Read in selected images dataset.

apiKey = hipp.dataquery.EE_login(input(), getpass())

explore_df = pd.read_csv('/home/elilouis/hsfm-geomorph/create_dems/baker_ee_image_dataset.csv')

# # Read in the geojson file with my EE image subset
import geopandas as gpd
subset_df = gpd.read_file(os.path.join(output_directory, 'ee_1950_subset_for_processing_with_nagap.geojson'))
explore_df = explore_df[explore_df.entityId.isin(subset_df.filename)]

# # Download images

output_directory = '/data2/elilouis/timesift/baker-ee/'
label     = 'test_download'

images_directory, calibration_reports_directory = hipp.dataquery.EE_downloadImages(
    apiKey,
    explore_df['entityId'].tolist(),
    label,
    output_directory
)

pd.set_option('display.max_columns', 100)

# # Correct the grid organization issue
#
# Just using cv2 to read and write the image does the trick

fixed_image_directory = images_directory.replace("raw_images", "raw_images_fixed")

# mkdir {fixed_image_directory}

import glob
import cv2
# this ignores files in deeper directories (ie the NAGAP imges)
files = glob.glob(os.path.join(images_directory, "*.tif"))
if not os.path.exists(images_directory.replace("raw_images", "raw_images_fixed")):
    os.makedirs(images_directory.replace("raw_images", "raw_images_fixed"))
print(f'Of {len(files)}, processing...', end=' ')
for i,file in enumerate(files):
    print(i+1, end=' ')
    x = cv2.imread(file)
    cv2.imwrite(file.replace("raw_images", "raw_images_fixed"), x)

# ls {fixed_image_directory}

# # Generate Fiducial Marker Proxy Templates

image_file = os.listdir(images_directory)[0]

fiducial_template_directory = os.path.join(output_directory, 'fiducials')
fiducial_template_directory

# +
hipp.core.create_fiducial_template(
    os.path.join(images_directory, image_file),
    fiducial_template_directory
    
)
# hipp.core.create_midside_fiducial_proxies_template(
#     '/data2/elilouis/generate_ee_dems_baker/raw_images/AR1LK0000010078.tif',
#     '/data2/elilouis/generate_ee_dems_baker/fiducials'
# )

# -
# # Detect Fiducial Marker Proxies with our new templates


preprocessed_images_directory = fixed_image_directory.replace('raw_images_fixed', 'preprocessed_images')
qc_directory = preprocessed_images_directory.replace("preprocessed_images", "preprocess_qc")

fixed_image_directory, preprocessed_images_directory

hipp.batch.preprocess_with_fiducial_proxies(
    fixed_image_directory,
    fiducial_template_directory,
    output_directory=preprocessed_images_directory,
    verbose=True,
    missing_proxy=None,
    qc_df=True,
    qc_df_output_directory=os.path.join(qc_directory, 'proxy_detection_data_frames'),
    qc_plots=True,
    qc_plots_output_directory=os.path.join(qc_directory, 'proxy_detection')
)

# # Create small thumbnails of processed images for quick checking

# !mogrify -format jpg -strip -interlace Plane -gaussian-blur 0.05 -quality 10% {os.path.join(preprocessed_images_directory, "*.tif")}

# # Create metashape metadata csv from EE results

explore_df.head(3)

new_df = pd.DataFrame()
new_df['image_file_name'] = explore_df['imageId'].apply(lambda x: f'{x}.tif')
new_df['lon'] = explore_df['centerLon']
new_df['lat'] = explore_df['centerLat']
new_df['alt'] = explore_df['altitudesFeet'] * 0.3048 # feet to meters
new_df['lon_acc'] = 1000.0
new_df['lat_acc'] = 1000.0
new_df['alt_acc'] = 1000.0
new_df['yaw'] = 0.0
new_df['pitch'] = 0.0
new_df['roll'] = 0.0
new_df['yaw_acc'] = 180.0
new_df['pitch_acc'] = 10.0
new_df['roll_acc'] = 10.0
new_df['focal_length'] = explore_df['focalLength']

metadata_file = os.path.join(output_directory, 'metashape_metadata_ee.csv')

new_df.to_csv(
    metadata_file,
    index=False
)

# # Do an initial SfM processing run just to check an orthoimage out

import hsfm

hsfm.metashape.authentication(license_path)

res = hsfm.metashape.images2las(
    'project',
    images_path=fixed_image_directory,
    images_metadata_file = metadata_file,
    output_path = os.path.join(output_directory, 'metashape_processing/'),
    image_matching_accuracy=4,
    densecloud_quality=4,
    pixel_pitch=0.025, #assumed value for earth explorer high res imagery,
)

# # Use `TimesiftPipeline` to process images

# ## Set paths for processing, a combined metashape metadata file, and a combined image metadata file

# Create an output path
timesift_output_path = os.path.join(output_directory, "mixed_timesift")
# !mkdir $output_path
combined_metadata_file_path = os.path.join(timesift_output_path, 'combined_metashape_metadata.csv')
combined_image_metadata_file_path = os.path.join(timesift_output_path, 'combined_image_metadata.csv')

# ## Copy all NAGAP and EE preprocessed images into a shared directory
#
# This is a waste of space and changes to Timesift/Pipeline could alleviate the need to do this

# Copy all preprocessed images into common directory
# should be at {timesift_output_path}/preprocessed_images
preprocessed_images_dir = os.path.join(output_directory, 'preprocessed_images')
# !mkdir $preprocessed_images_dir


# !cp -a {ee_preprocessed_images_path}/. {preprocessed_images_dir}/
# !cp -a {nagap_preprocessed_images_path}/. {preprocessed_images_dir}/

# !ls $ee_preprocessed_images_path | wc -l
# !ls $nagap_preprocessed_images_path | wc -l
# !ls $preprocessed_images_dir | wc -l

# ## Make a combined metashape metadata file and set the pixel pitches appropriately

# +
# Make a combined metadata file and make sure a pixel_pitch column is set (0.02 for nagap, 0.025 for EE)
ee_metadata_file = metadata_file


ee_metadata_df = pd.read_csv(ee_metadata_file)
#why do i need to add AR here?
ee_metadata_df['image_file_name'] = 'AR' + ee_metadata_df['image_file_name']

ee_metadata_df['pixel_pitch'] = 0.025
nagap_metadata_df = pd.read_csv(nagap_metadata_file)
nagap_metadata_df['pixel_pitch'] = 0.02

combined_metadata_df = pd.concat([ee_metadata_df, nagap_metadata_df])
combined_metadata_df.to_csv(combined_metadata_file_path, index=False)
# -

# ## Create a image info dataframe (contains file name and date, like one that would come from using HIPP to querying NAGAP images)
#
# See the documentation for TimesiftPipeline for the argument "image_metadata_file".

# it needs these columns (only) [["fileName", "Year", "Month", "Day"]] and the fileName column should NOT end with .tif
# create this by getting NAGAP image info using the large-nagap data csv that comes with hipp 
# and ee_metadata_df, with information provided for the 4 columns mentioned above added in manually
nagap_image_info_df = pd.read_csv(nagap_image_info_fn)


nagap_metadata_df['fileName'] = nagap_metadata_df['image_file_name'].apply(lambda x: x.replace('.tif', ''))
nagap_image_info_df = nagap_metadata_df.merge(nagap_image_info_df, on=['fileName'])[["fileName", "Year", "Month", "Day"]]
nagap_image_info_df.head(1)

ee_metadata_df.loc[:, 'fileName'] = ee_metadata_df.loc[:, 'image_file_name'].apply(lambda x: x.replace('.tif', ''))
ee_image_info_df = ee_metadata_df[['fileName']]

# Assign dates to correct data (EE LK000 images with numbers in the 10,000s are actually 08.17,1950 and in the 20,000s are actually 09.02.1950)
ee_image_info_df.loc[:, 'Year'] = '50'
ee_image_info_df.loc[:, 'Month'] = '09'
ee_image_info_df.loc[:, 'Day'] = '02'

combined_image_info_df = pd.concat([nagap_image_info_df, ee_image_info_df])

combined_image_info_df.head()

combined_image_info_df.to_csv(combined_image_metadata_file_path, index=False)

# # Run Timesift, step by step

from hsfm.pipeline import TimesiftPipeline

# Can use these if you did not run stuff earier in the notebook
# combined_metadata_file_path = '/data2/elilouis/timesift/baker-ee/mixed_timesift/combined_metashape_metadata.csv'
# combined_image_metadata_file_path = '/data2/elilouis/timesift/baker-ee/mixed_timesift/combined_image_metadata.csv'
# preprocessed_images_dir = '/data2/elilouis/timesift/baker-ee/preprocessed_images'
# timesift_output_path = '/data2/elilouis/timesift/baker-ee/mixed_timesift'

# # Create the timesift pipeline with proper paths in advance
timesift_pipeline = TimesiftPipeline(
    metashape_metadata_file = combined_metadata_file_path,
    image_metadata_file = combined_image_metadata_file_path,
    raw_images_directory = preprocessed_images_dir,
    output_directory = timesift_output_path,
    reference_dem_lowres = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_10m.tif',
    reference_dem_hires = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif',
    densecloud_quality = 2,
    image_matching_accuracy = 1,
    output_DEM_resolution = 1,
    license_path="/home/elilouis/hsfm/uw_agisoft.lic",
    parallelization=1
)

metadata_timesift_aligned_file, unaligned_cameras_df = timesift_pipeline._generate_multi_epoch_densecloud()
# Do something with the unaligned cameras!!!
_ = timesift_pipeline._save_image_footprints()
_ = timesift_pipeline._export_camera_calibration_files()
_ = timesift_pipeline._prepare_single_date_data(metadata_timesift_aligned_file)
dict_of_subsets_by_date = timesift_pipeline._find_clusters_in_individual_clouds()
_ = timesift_pipeline._generate_subsets_for_each_date(dict_of_subsets_by_date)

# _ = timesift_pipeline._process_individual_clouds(['50_9.0'], output_DEM_resolution_override=5)
# _ = timesift_pipeline._process_individual_clouds(['????'], output_DEM_resolution_override=1)
