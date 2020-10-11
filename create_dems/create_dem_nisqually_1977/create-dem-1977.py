# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
# -

data_dir = '/home/elilouis/hsfm-geomorph/data'
hsfm_dir = '/home/elilouis/hsfm'
license_path = '/home/elilouis/hsfm/uw_agisoft.lic'

hsfm.metashape.authentication(license_path)

# +
targets_file = '../../identify-imagery/targets_nisqually_1977.csv'
image_directory = 'input_data/images_source'
preprocess_output_directory= 'input_data/images_preprocessed'

fiducial_template_dir = f'{data_dir}/fiducials_nisqually_1977'

image_subset_df = pd.read_csv(targets_file)
# -


# # Pre-process

# ## Download thumbnails

# + tags=[]
# image_type options: pid_tiff (high res), pid_tn (low res), pid_jpeg (high res)
hsfm.batch.download_images_to_disk(targets_file, output_directory=image_directory, image_type='pid_tiff')
# -

imgplot = plt.imshow(mpimg.imread(f'{image_directory}/NAGAP_77V6_263.tif'))
plt.gcf().set_size_inches(5,5)
plt.show()

# ## Create fiducial marker templates
# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

# + tags=[]
# ls $fiducial_template_dir
# -

hsfm.utils.create_fiducials(img, output_directory=fiducial_template_dir)

# ## Preprocess Step 1 - detect fiducial markers, crop, enhance contrast
# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

# + tags=[]
hsfm.batch.preprocess_images(fiducial_template_dir,
                             camera_positions_file_name=targets_file,
                             image_directory=image_directory,
                             output_directory=preprocess_output_directory,
                             qc=True)
# -

# ## Open Reference DEMs (High-res WA-DNA LIDAR and low-res SRTM)
# These were proccessed in adjascent notebook `prepare-reference-dems.ipynb`

# ### Low-Res DEM 

srtm_reference_dem = f'{data_dir}/reference_dem/SRTM3/SRTM3.vrt'
srtm_reference_dem_warped = f'{data_dir}/reference_dem/SRTM3/SRTM3_warped.tif'

# !gdalwarp -t_srs EPSG:4326 $srtm_reference_dem $srtm_reference_dem_warped

# !gdalinfo $srtm_reference_dem_warped

# !gdalinfo $srtm_reference_dem

hsfm.plot.plot_dem_from_file(srtm_reference_dem_warped)

# ## Preprocess Step 2 - calculate new csv file with heading info

# Generate input for Metashape

hsfm.core.prepare_metashape_metadata(targets_file,
                                     reference_dem=srtm_reference_dem_warped)

# ### High-Res DEM 

reference_dem = f'{data_dir}/reference_dem_highres/reference_dem_final-adj.tif'

hsfm.plot.plot_dem_from_file(reference_dem)

# ## Run Processing with Metashape
#
# Reassign the reference DEM
#
# Use the high resolution DEM in EPSG 32610.

# +
image_matching_accuracy = 4
densecloud_quality      = 4

project_name          = 'nisqually_1977'
input_path            = 'input_data'
output_path           = 'metashape/'
images_path           = 'input_data/images_preprocessed/'
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
# -

hsfm.plot.plot_dem_from_file(clipped_reference_dem)

hsfm.plot.plot_dem_from_file(aligned_dem_file)

# # Create DEM of Difference

dem_difference = hsfm.utils.difference_dems(clipped_reference_dem, aligned_dem_file, verbose=True)

hsfm.plot.plot_dem_from_file(dem_difference)

# ## Mask Glaciers of DEM of Difference

import geopandas as gpd
import fiona
import rasterio
import rasterio.mask

aois_gdf = gpd.read_file(f'{data_dir}/aois.geojson')
rainier_polygon = aois_gdf[aois_gdf.name == 'Mt. Rainier']

glacier_gdf = gpd.read_file(f'{data_dir}/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp')

rainier_glaciers_gdf = gpd.sjoin(glacier_gdf, rainier_polygon)

rainier_glaciers_gdf.plot()

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
