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

# +
import bare
import hsfm
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# +
targets_file = 'targets_kautz_watershed_1977.csv'
image_directory = 'input_data/images_source'
preprocess_output_directory= 'input_data/images_preprocessed'

fiducial_template_dir = 'input_data/fiducials'

image_subset_df = pd.read_csv('targets_kautz_watershed_1977.csv')
# -


# # Pre-process

# ## Download thumbnails

# image_type options: pid_tiff (high res), pid_tn (low res), pid_jpeg (high res)
hsfm.batch.download_images_to_disk(targets_file, output_directory=image_directory, image_type='pid_tiff') 

imgplot = plt.imshow(mpimg.imread(f'{image_directory}/NAGAP_77V6_263.tif'))
plt.gcf().set_size_inches(5,5)
plt.show()

# ## Create fiducial marker templates
# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

hsfm.utils.create_fiducials(img, output_directory=fiducial_template_dir)

# ## Preprocess Step 1 - detect fiducial markers, crop, enhance contrast
# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

hsfm.batch.preprocess_images(fiducial_template_dir,
                             camera_positions_file_name=targets_file,
                             image_directory=image_directory,
                             output_directory=preprocess_output_directory,
                             qc=True)

# ## Open Reference DEMs (High-res WA-DNA LIDAR and low-res SRTM)
# These were proccessed in adjascent notebook `prepare-reference-dems.ipynb`

# ### Low-Res DEM 

# Why use VRT file here instead of a tiff?

srtm_reference_dem = 'input_data/reference_dem/SRTM3/SRTM3.vrt'

# ## Preprocess Step 2 - calculate new csv file with heading info

hsfm.batch.calculate_heading_from_metadata(image_subset_df, 
                                          output_directory=preprocess_output_directory,
                                          reference_dem=srtm_reference_dem,
                                          for_metashape=True)

# ### High-Res DEM 

reference_dem = 'input_data/reference_dem_highres/reference_dem_final-adj.tif'

hsfm.plot.plot_dem_from_file(reference_dem)

# ## Run Processing with Metashape
#
# Reassign the reference DEM
#
# Use the high resolution DEM in EPSG 32610.

# +
image_matching_accuracy = 4
densecloud_quality      = 4

project_name          = 'kautz'
input_path            = 'input_data'
output_path           = 'metashape/'
images_path           = 'input_data/images_preprocessed/'
images_metadata_file  = 'input_data/images_preprocessed/metashape_metadata.csv'
focal_length          = 152
pixel_pitch           = 0.02
verbose               = True
rotation_enabled      = True
# -

# Try changing data in images_metadata_file  = 'input_data/metashape_metadata.csv':
#
# yaw/pitch/roll = 0
#
# yaw_acc = 180
#
# pitch_acc/roll_acc = 10

hsfm.metashape.authentication('/home/elilouis/hsfm/uw_agisoft.lic')

project_file, point_cloud_file = hsfm.metashape.images2las(project_name,
                                            images_path,
                                            images_metadata_file,
                                            output_path,
                                            focal_length            = focal_length,
                                            pixel_pitch             = pixel_pitch,
                                            image_matching_accuracy = image_matching_accuracy,
                                            densecloud_quality      = densecloud_quality,
                                            rotation_enabled        = rotation_enabled)


epsg_code = 'EPSG:'+ hsfm.geospatial.get_epsg_code(reference_dem)
dem = hsfm.asp.point2dem(point_cloud_file, 
                         '--nodata-value','-9999',
                         '--tr','0.5',
                         '--threads', '10',
                         '--t_srs', epsg_code,
                         verbose=verbose)

clipped_reference_dem = 'metashape/reference_dem_clip.tif'
clipped_reference_dem = hsfm.utils.clip_reference_dem(dem, 
                                                      reference_dem,
                                                      output_file_name = clipped_reference_dem,
                                                      buff_size        = 2000,
                                                      verbose = verbose)

aligned_dem_file, _ =  hsfm.asp.pc_align_p2p_sp2p(dem,
                                                  clipped_reference_dem,
                                                  output_path,
                                                  verbose = verbose)

# + jupyter={"outputs_hidden": true}
hsfm.utils.dem_align_custom(clipped_reference_dem,
                            aligned_dem_file,
                            output_path,
                            verbose = verbose)

# + jupyter={"outputs_hidden": true}
hsfm.plot.plot_dem_from_file(clipped_reference_dem)

# + jupyter={"outputs_hidden": true}
hsfm.plot.plot_dem_from_file(aligned_dem_file)
# -

# # Create DEM of Difference

dem_difference = hsfm.utils.difference_dems(clipped_reference_dem, aligned_dem_file, verbose=True)

hsfm.plot.plot_dem_from_file(dem_difference)

# ## Mask Glaciers of DEM of Difference

import geopandas as gpd

aois_gdf = gpd.read_file('/home/elilouis/hsfm-geomorph/aois.geojson')
rainier_polygon = aois_gdf[aois_gdf.name == 'Mt. Rainier']

glacier_gdf = gpd.read_file('/home/elilouis/02_rgi60_WesternCanadaUS.shp')

rainier_glaciers_gdf = gpd.sjoin(glacier_gdf, rainier_polygon)

rainier_glaciers_gdf.plot()

import fiona
import rasterio
import rasterio.mask

# change glacier gdf crs to that of the raster

rainier_glaciers_gdf = rainier_glaciers_gdf.to_crs(epsg='32610')

dem_difference_masked = dem_difference.replace(".tif", "-masked.tif")
dem_difference_masked

# +
with rasterio.open(dem_difference) as src:
    out_image, out_transform = rasterio.mask.mask(src, rainier_glaciers_gdf.geometry, invert=True)
    out_meta = src.meta
    
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(dem_difference_masked, "w", **out_meta) as dest:
    dest.write(out_image)
# -

hsfm.plot.plot_dem_from_file(dem_difference_masked)

dem_difference_masked


