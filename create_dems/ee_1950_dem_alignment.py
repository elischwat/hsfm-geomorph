# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Testing different options of pc_align to improve the 1950 EE DEM.
#
# When I ran the pipeline, I was using a DEM with limited coverage (the WA DNR DEM, I think)
#
# Now, I try to align the point clouds using the Copernicus DEM that has coverage for nearly all of the area covered by the EE DEM. Make sure that the denser dataset (the 1950 SfM dataset) is used as the reference dataset for pc_align, and then make sure to use the inverse transform.

PROJECT_PATH = "/data2/elilouis/hsfm-geomorph/data/create_dems/ee_1950_dem_alignment/"
import os
def resource_path():
    return PROJECT_PATH
def resource(obj):
    return os.path.join(PROJECT_PATH, obj)
def resources():
    return os.listdir(PROJECT_PATH)
from rasterio.enums import Resampling

import rioxarray as rix
from shapely.geometry import box
import geopandas as gpd
import contextily as ctx

resources()

# # Prep data

# ## Load data

# +
reference_dem_fn = (
    "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/copernicus" +
    "/baker_copernicus_reference_dem.tif"
)

sfm_dem_aligned_fn = (
    "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/" +
    "50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+4.02_y+6.26_z-1.88_align.tif"
)

sfm_dem_fn = (
    "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/" +
    "50_9.0/cluster0/1/project-DEM.tif"
)

# +
reference_dem = rix.open_rasterio(
    reference_dem_fn, 
    masked=True
).squeeze()

sfm_dem = rix.open_rasterio(
    sfm_dem_fn,
    masked=True
).squeeze()

sfm_dem_aligned = rix.open_rasterio(
    sfm_dem_aligned_fn,
    masked=True
).squeeze()
# -

[r.rio.crs for r in [reference_dem, sfm_dem, sfm_dem_aligned]]

# + tags=[]
geoseries = gpd.GeoSeries([
    box(*reference_dem.rio.reproject(sfm_dem.rio.crs).rio.bounds()),
    box(*sfm_dem.rio.bounds()),
    box(*sfm_dem_aligned.rio.bounds()),
], crs='EPSG:32610').plot(alpha=0.3)

# + [markdown] tags=[]
# ## Crop reference dem to the buffered boundsof the SfM DEMs
# Note that they are in different projections and we do this before reprojecting one dataset so that the reprojection can process more quickly.
# -

BUFFER_METERS = 4000
reference_dem = reference_dem.rio.clip([ 
    box(*sfm_dem.rio.bounds()).buffer(BUFFER_METERS)
])

geoseries = gpd.GeoSeries([
    box(*reference_dem.rio.reproject(sfm_dem.rio.crs).rio.bounds()),
    box(*sfm_dem.rio.bounds()),
    box(*sfm_dem_aligned.rio.bounds()),
], crs='EPSG:32610').plot(alpha=0.3)

# ## Save modified reference_dem

reference_dem.rio.to_raster(resource('reference_dem.tif'))

resources()

# ## Copy original SfM DEM into working directory

# !echo {sfm_dem_fn}
# !echo {resource('sfm_dem.tif')}

# cp {sfm_dem_fn} {resource('sfm_dem.tif')}

# # Test pc_align 

[r.rio.resolution() for r in [sfm_dem, reference_dem]]

# ## Run pc_align (point-to-plane)
#
# Make sure the denser DEM is the "reference" dem, which is the (historical) SfM DEM in this case. Then use the output reverse transform to move the incorrectly aligned SfM DEM.

# + jupyter={"outputs_hidden": true} tags=[]
# !pc_align \
#     {resource('sfm_dem.tif')} \
#     {resource('reference_dem.tif')} \
#     -o {os.path.join(resource_path(), 'point-to-plane/')} \
#     --max-displacement 5000 \
#     --alignment-method=point-to-plane
# -

# Apply the transform (the inverse one). Note that switch the order of the DEMs here (because we are using the inverse transform)

# + jupyter={"outputs_hidden": true} tags=[]
# !pc_align \
#     {resource('reference_dem.tif')} \
#     {resource('sfm_dem.tif')} \
#     -o {os.path.join(resource_path(), 'point-to-plane', 'run/r')} \
#     --max-displacement -1 \
#     --alignment-method=point-to-plane \
#     --initial-transform {resource("point-to-plane/-inverse-transform.txt")} \
#     --num-iterations 0 \
#     --save-transformed-source-points
# -

# Generate a DEM from the wierd point cloud tif

# !point2dem {os.path.join(resource_path(), 'point-to-plane', 'run/-trans_source.tif')}

# ## Run pc_align (point-to-point)

# !pc_align \
#     {resource('sfm_dem.tif')} \
#     {resource('reference_dem.tif')} \
#     -o {os.path.join(resource_path(), 'point-to-point/')} \
#     --max-displacement 5000 \
#     --alignment-method=point-to-point

# !pc_align \
#     {resource('reference_dem.tif')} \
#     {resource('sfm_dem.tif')} \
#     -o {os.path.join(resource_path(), 'point-to-point', 'run/')} \
#     --max-displacement -1 \
#     --alignment-method=point-to-point \
#     --initial-transform {resource("point-to-point/-inverse-transform.txt")} \
#     --num-iterations 0 \
#     --save-transformed-source-points

# !point2dem {os.path.join(resource_path(), 'point-to-point', 'run/-trans_source.tif')}

# ## Run pc_align (similarity-point-to-point)

# !pc_align \
#     {resource('sfm_dem.tif')} \
#     {resource('reference_dem.tif')} \
#     -o {os.path.join(resource_path(), 'similarity-point-to-point/')} \
#     --max-displacement 5000 \
#     --alignment-method=similarity-point-to-point

# !pc_align \
#     {resource('reference_dem.tif')} \
#     {resource('sfm_dem.tif')} \
#     -o {os.path.join(resource_path(), 'similarity-point-to-point', 'run/')} \
#     --max-displacement -1 \
#     --alignment-method=similarity-point-to-point \
#     --initial-transform {resource("similarity-point-to-point/-inverse-transform.txt")} \
#     --num-iterations 0 \
#     --save-transformed-source-points

# !point2dem {os.path.join(resource_path(), 'similarity-point-to-point', 'run/-trans_source.tif')}

# # Test hsfm.asp.

import hipp

# +

# !which python
# -


