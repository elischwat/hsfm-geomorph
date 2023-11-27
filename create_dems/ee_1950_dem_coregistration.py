# -*- coding: utf-8 -*-
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

# %load_ext autoreload
# %autoreload 2

# +
from pathlib import Path

import rioxarray as rix
from rioxarray import merge
from rasterio.plot import show
import numpy as np
import xdem as du
# -

# # Look at EE 1950 DEM

ee_dem_fn = '/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/project-DEM.tif'

ee_dem = rix.open_rasterio(ee_dem_fn, masked=True)

ee_dem.squeeze().plot.imshow()

# # Download two Copernicus DEM tiles

ref_dem_dir = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/'
wadnr_ref_dem_1m = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_1m.tif'
cop_ref_dem_1m = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/copernicus_aligned_01m.tif'

cog1 = 's3://copernicus-dem-30m/Copernicus_DSM_COG_10_N48_00_W122_00_DEM/Copernicus_DSM_COG_10_N48_00_W122_00_DEM.tif'
cog2 = 's3://copernicus-dem-30m/Copernicus_DSM_COG_10_N48_00_W123_00_DEM/Copernicus_DSM_COG_10_N48_00_W123_00_DEM.tif'
# ! aws s3 cp {cog1} --no-sign-request {ref_dem_dir}
# ! aws s3 cp {cog2} --no-sign-request {ref_dem_dir}

# # Join the two Copernicus DEM tiles

# +
elements = [
    rix.open_rasterio(
        Path("/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/") / Path(cog1).name
    ),
    rix.open_rasterio(
        Path("/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/") / Path(cog2).name
    )
]
merged = merge.merge_arrays(elements, nodata=0.0)

image = merged.values
# -

show(image)

copernicus_dem_fn = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/copernicus_joined_dem.tif"

merged.rio.to_raster(copernicus_dem_fn)

# # Difference Copernicus and EE DEMs

# ## Decrease resolution of the EE DEM

ee_dem = ee_dem.rio.reproject(ee_dem.rio.crs, resolution=(30,30))

# ## Reproject/match copernicus DEM to the EE DEM

copernicus_dem = rix.open_rasterio(copernicus_dem_fn, masked=True)

copernicus_dem = copernicus_dem.rio.reproject_match(ee_dem)

ee_dem.squeeze().plot.imshow(cmap='viridis', vmin=0, vmax=3500)

copernicus_dem.squeeze().plot.imshow(cmap='viridis', vmin=0, vmax=3500)

# ## Subtract DEMs

diff_dem = copernicus_dem - ee_dem

diff_dem.plot.hist(bins=50)

diff_dem.squeeze().plot.imshow(cmap='PuOr')

# # Coregister DEMs

copernicus_dem.rio.to_raster('copernicus_dem.tif')
ee_dem.rio.to_raster('ee_dem.tif')

# mkdir coregistration

# + jupyter={"outputs_hidden": true}
# !dem_align.py \
# -max_offset 1000 \
# -mode 'nuth' \
# -mask_list glaciers nlcd \
# -outdir  coregistration \
# copernicus_dem.tif ee_dem.tif
# -

aligned_dem_fn = "ee_dem_dem_align/ee_dem_copernicus_dem_nuth_x+3.25_y+71.41_z+328.70_align.tif"
aligned_dem_masked_fn = "ee_dem_dem_align/ee_dem_copernicus_dem_nuth_x+3.25_y+71.41_z+328.70_align_filt.tif"

















aligned_dem_fn = "coregistration/*align.tif"
aligned_dem_masked_fn = "coregistration/*align_filt.tif"

# # Deramp the SfM DEM

import geoutils as gu

reference_dem = gu.georaster.Raster('copernicus_dem.tif')
dem = gu.georaster.Raster(aligned_dem_fn)
masked_dem = gu.georaster.Raster(aligned_dem_masked_fn)

import matplotlib.pyplot as plt
import pathlib

fig,ax = plt.subplots(1,3,figsize=(15,5))
dem.show(ax=ax[0])
masked_dem.show(ax=ax[1])
reference_dem.show(ax=ax[2])
[axi.set_axis_off() for axi in ax.ravel()];

masked_dem    = masked_dem.reproject(dem, nodata=masked_dem.nodata)
reference_dem = reference_dem.reproject(dem, nodata=reference_dem.nodata)


dem_array           = dem.data.squeeze().copy()
masked_dem_array    = masked_dem.data.squeeze().copy()
reference_dem_array = reference_dem.data.squeeze().copy()


dem.crs == masked_dem.crs == reference_dem.crs

dem_array.shape == masked_dem_array.shape == reference_dem_array.shape


def mask_array_with_nan(array,nodata_value):
    """
    Replace dem nodata values with np.nan.
    """
    mask = (array == nodata_value)
    masked_array = np.ma.masked_array(array, mask=mask)
    masked_array = np.ma.filled(masked_array, fill_value=np.nan)
    
    return masked_array


dem_array           = mask_array_with_nan(dem_array,dem.nodata)
masked_dem_array    = mask_array_with_nan(masked_dem_array,masked_dem.nodata)
reference_dem_array = mask_array_with_nan(reference_dem_array,reference_dem.nodata)

fig,ax = plt.subplots(1,3,figsize=(15,5))
ax[0].imshow(dem_array)
ax[1].imshow(masked_dem_array)
ax[2].imshow(reference_dem_array)
[axi.set_axis_off() for axi in ax.ravel()];


elevation_difference_array = reference_dem_array - masked_dem_array

nmad_before = du.spatial_tools.nmad(elevation_difference_array)
print(nmad_before)

# +

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(elevation_difference_array, cmap='RdBu',)
fig.colorbar(im,extend='both')
ax.set_axis_off();
# -

x_coordinates, y_coordinates = np.meshgrid(
    np.arange(elevation_difference_array.shape[1]),
    np.arange(elevation_difference_array.shape[0])
)

ramp = du.coreg.deramping(elevation_difference_array, x_coordinates, y_coordinates, 2)


# +
mask_array = np.zeros_like(elevation_difference_array)
mask_array += ramp(x_coordinates, y_coordinates)

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(mask_array, cmap='RdBu',)
fig.colorbar(im,extend='both')
ax.set_title('estimated correction surface')
ax.set_axis_off();

# -

# # Correct masked DEM¶
#

masked_dem_array_corrected = masked_dem_array.copy()
masked_dem_array_corrected += ramp(x_coordinates, y_coordinates)

# +

elevation_difference_array_corrected = reference_dem_array - masked_dem_array_corrected
# -

nmad_after = du.spatial_tools.nmad(elevation_difference_array_corrected)
print(nmad_after)

fig,ax = plt.subplots(1,2,figsize=(20,20))
im0 = ax[0].imshow(elevation_difference_array, cmap='RdBu',clim=(-2, 2))
# fig.colorbar(im,extend='both')
ax[0].set_title('NMAD before: '+ f"{nmad_before:.3f} m")
ax[0].set_axis_off();
im1 = ax[1].imshow(elevation_difference_array_corrected, cmap='RdBu',clim=(-2, 2))
# fig.colorbar(im1,extend='both')
ax[1].set_title('NMAD after: '+ f"{nmad_after:.3f} m")
ax[1].set_axis_off();


dem_array += ramp(x_coordinates, y_coordinates)

elevation_difference_dem_array = reference_dem_array - dem_array


fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(elevation_difference_dem_array, cmap='RdBu')
fig.colorbar(im,extend='both')
ax.set_axis_off();

# should probably change the no data value back and assign more efficient dtype
corrected_dem = gu.georaster.Raster.from_array(
    data=elevation_difference_dem_array,
    transform=dem.transform,
    crs=dem.crs,
    nodata=np.nan
)
out_fn = 'ee_dem_final.tif'
corrected_dem.save(out_fn)


