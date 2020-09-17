# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import bare
import hsfm
import pandas as pd

targets_file = 'targets_kautz_watershed_1977.csv'
image_directory = 'input_data/thumbnails'
output_directory= 'input_data/images'


# # Pre-process imagery

# ## Download thumbnails

# cat $targets_file | head -2

# + jupyter={"outputs_hidden": true}
hsfm.batch.download_images_to_disk(targets_file, 
                                   output_directory=image_directory,
                                   image_type='pid_tiff') # pid_tiff, pid_tn, pid_jpeg
# -

# ## Examine a thumbnail

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('input_data/thumbnails/NAGAP_77V6_263.tif')
imgplot = plt.imshow(img)
plt.gcf().set_size_inches(20,20)
plt.xlim
plt.show()

# Looks like the fiducal markers don't match an existing pattern...

# ## Create fiducial marker templates

# + jupyter={"outputs_hidden": true}
hsfm.utils.create_fiducials(img)
# -

fiducial_template_dir = 'fiducials'


# ## Detect fiducial markers, crop, and enhance contrast

hsfm.batch.preprocess_images(fiducial_template_dir,
                             camera_positions_file_name=targets_file,
                             image_directory=image_directory,
                             output_directory=output_directory,
                             qc=True)

# # Prepare Reference DEM

# ## Open up high-res LIDAR reference DEM

# Get corner coordinates from a high resolution LIDAR DEM that is cropped to Mt. Rainier so I can download a coarse SRTM reference DEM

reference_dem_high_res_file = '/Users/elischwat/Development/dem_differencing/all_dem_dem_differencing/2007_final.tif'

hsfm.plot.plot_dem_from_file(reference_dem_high_res_file)

# Convert to meters

reference_dem_high_res_file_in_meters = 'input_data/reference_dem_highres/reference_dem_m.tiff'
reference_dem_high_res_file_in_meters_warped = 'input_data/reference_dem_highres/reference_dem_m_epsg32610.tiff'

# !ls input_data/reference_dem_highres
# !mkdir input_data/reference_dem_highres

# Use gdal_calc to convert from feet to meters

# + jupyter={"outputs_hidden": true}
# !gdal_calc.py --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=IF_SAFER --NoDataValue=-9999 --calc 'A*0.3048' -A $reference_dem_high_res_file --outfile $reference_dem_high_res_file_in_meters
# -

hsfm.plot.plot_dem_from_file(reference_dem_high_res_file_in_meters)

# Use gdalwarp to convert from 26710 (NAD27 UTM 10N) -> 32610 (WGS84 UTM 10N)

# !gdalinfo $reference_dem_high_res_file_in_meters | head -5 | tail -1

# ### LAST I TRIED THIS, I HAD TO DO IT THROUGH QGIS TO GET IT WORK...

# !gdalwarp -t_srs EPSG:32610 -r near -of GTiff $reference_dem_high_res_file_in_meters $reference_dem_high_res_file_in_meters_warped

hsfm.plot.plot_dem_from_file(reference_dem_high_res_file_in_meters_warped)

# !dem_geoid  --reverse-adjustment $reference_dem_high_res_file_in_meters_warped -o 'input_data/reference_dem_highres/reference_dem_final'

reference_dem_high_res_file_final = 'input_data/reference_dem_highres/reference_dem_final-adj.tif'

hsfm.plot.plot_dem_from_file(reference_dem_high_res_file_final)

# !gdalinfo $reference_dem_high_res_file_final | grep "Corner Coord" --after 5

# Convert using this tool https://www.rapidtables.com/convert/number/degrees-minutes-seconds-to-degrees.html

LLLON = -121.948
LLLAT = 46.74643
URLON = -121.6145
URLAT = 47.00322

# ## Download coarse SRTM reference DEM

reference_dem = hsfm.utils.download_srtm(LLLON,
                                         LLLAT,
                                         URLON,
                                         URLAT,
                                         output_directory='input_data/reference_dem/',
                                         verbose=False)

hsfm.plot.plot_dem_from_file(reference_dem)


# ## Generate input for Metashape

reference_dem = 'input_data/reference_dem/SRTM3/cache/srtm.vrt'
output_directory = 'input_data'

# ls

df = pd.read_csv('targets_kautz_watershed_1977.csv')

hsfm.batch.calculate_heading_from_metadata(df, 
                                          output_directory=output_directory,
                                          reference_dem=reference_dem,
                                          for_metashape=True)

# ls -la input_data/metashape_metadata.csv

hsfm.metashape.authentication('/Users/elischwat/Development/hsfm/metashape_trial.lic')

# ## Run Processing with Metashape

# +
image_matching_accuracy = 4
densecloud_quality      = 4

project_name          = 'kautz'
input_path            = 'input_data'
output_path           = 'metashape/'
images_path           = 'input_data/thumbnails/'
images_metadata_file  = 'input_data/metashape_metadata.csv'
focal_length          = 152
pixel_pitch           = 0.02
verbose               = True
rotation_enabled      = True
# -

project_file, point_cloud_file = hsfm.metashape.images2las(project_name,
                                            images_path,
                                            images_metadata_file,
                                            output_path,
                                            focal_length            = focal_length,
                                            pixel_pitch             = pixel_pitch,
                                            image_matching_accuracy = image_matching_accuracy,
                                            densecloud_quality      = densecloud_quality,
                                            rotation_enabled        = rotation_enabled)


