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
import xdem as du
import geoutils as gu

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
# -

# # Load DEMs

dem_fn = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/testing/pc_align_p2p_sp2p/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_nuth_x+130.61_y-135.81_z+140.13_align.tif"
masked_dem_fn = dem_fn.replace("align.tif", "align_filt.tif")
reference_dem_fn = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/copernicus_joined_dem.tif"

dem = gu.georaster.Raster(dem_fn)
masked_dem = gu.georaster.Raster(masked_dem_fn)
reference_dem = gu.georaster.Raster(reference_dem_fn)

fig,ax = plt.subplots(1,3,figsize=(15,5))
dem.show(ax=ax[0],title='SfM DEM')
masked_dem.show(ax=ax[1],title='SfM DEM, Masked')
reference_dem.show(ax=ax[2],title='LIDAR Reference DEM')
[axi.set_axis_off() for axi in ax.ravel()];

# # Reproject and warp to common dims

dem = dem.reproject(dem, nodata=dem.nodata)
masked_dem = masked_dem.reproject(masked_dem, nodata=dem.nodata)
reference_dem = reference_dem.reproject(dem, nodata=reference_dem.nodata)

# # Extract np.ndarray

dem_array = dem.data.squeeze().copy()
masked_dem_array = masked_dem.data.squeeze().copy()
reference_dem_array = reference_dem.data.squeeze().copy()

# # Validate

dem.crs == masked_dem.crs == reference_dem.crs

dem_array.shape == masked_dem_array.shape == reference_dem_array.shape


# # Mask fill values with np.nan

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
ax[0].set_title('SfM DEM')
ax[1].imshow(masked_dem_array)
ax[1].set_title('SfM DEM, Masked')
ax[2].imshow(reference_dem_array)
ax[2].set_title('LIDAR Reference DEM')
[axi.set_axis_off() for axi in ax.ravel()];


# # Initial difference

elevation_difference_array = reference_dem_array - masked_dem_array
# elevation_difference_array = reference_dem_array - dem_array

nmad_before = du.spatial_tools.nmad(elevation_difference_array)
print(nmad_before)

# # Plot

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(
    elevation_difference_array, 
    cmap='RdBu',
#     clim=(-2, 2)
)
fig.colorbar(im,extend='both')
ax.set_title('Elevation diff in meters, masked')
ax.set_axis_off();

# # Fit surface

x_coordinates, y_coordinates = np.meshgrid(
    np.arange(elevation_difference_array.shape[1]),
    np.arange(elevation_difference_array.shape[0])
)


ramp = du.coreg.deramping(elevation_difference_array, x_coordinates, y_coordinates, 2)

# +
mask_array = np.zeros_like(elevation_difference_array)
mask_array += ramp(x_coordinates, y_coordinates)

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(
    mask_array, 
    cmap='RdBu',
#     clim=(-2, 2)
)
fig.colorbar(im,extend='both')
ax.set_title('estimated correction surface')
ax.set_axis_off();
# -
# # Correct DEM


dem_array_corrected = dem_array.copy()
dem_array_corrected += ramp(x_coordinates, y_coordinates)
elevation_difference_array_corrected = reference_dem_array - dem_array_corrected

nmad_after = du.spatial_tools.nmad(elevation_difference_array_corrected)
print(nmad_after)

fig,ax = plt.subplots(1,2,figsize=(20,20))
im0 = ax[0].imshow(elevation_difference_array, cmap='RdBu', clim=(-400,400))
fig.colorbar(im)
ax[0].set_title('NMAD before: '+ f"{nmad_before:.3f} m")
ax[0].set_axis_off();
im1 = ax[1].imshow(elevation_difference_array_corrected, cmap='RdBu', clim=(-400,400))
fig.colorbar(im1)
ax[1].set_title('NMAD after: '+ f"{nmad_after:.3f} m")
ax[1].set_axis_off();

# # Apply correction to unmasked DEM

# +
dem_array += ramp(x_coordinates, y_coordinates)

# should probably change the no data value back and assign more efficient dtype
corrected_dem = gu.georaster.Raster.from_array(
    data=dem_array_corrected,
    transform=dem.transform,
    crs=dem.crs,
    nodata=np.nan
)
out_fn = dem_fn.replace(".tif", "_corrected.tif")
corrected_dem.save(out_fn)
# -

out_fn


