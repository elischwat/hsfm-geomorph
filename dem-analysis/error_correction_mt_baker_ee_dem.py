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

import hsfm
import xdem as du
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt

dem_fn = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+4.02_y+6.26_z-1.88_align.tif"
diff_fn = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+4.02_y+6.26_z-1.88_align_diff.tif"
dem_masked_fn = "/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+4.02_y+6.26_z-1.88_align_filt.tif"
dem_reference_fn = ("/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_5m.tif")

# +
dem = gu.georaster.Raster(dem_fn)
ref_dem = gu.georaster.Raster(dem_reference_fn)

dem = dem.reproject(ref_dem)
dem.crop(ref_dem)
# -

dem.show(), ref_dem.show()

print(dem), print(ref_dem)


def mask_array_with_nan(array,nodata_value):
    """
    Replace dem nodata values with np.nan.
    """
    mask = (array == nodata_value)
    masked_array = np.ma.masked_array(array, mask=mask)
    masked_array = np.ma.filled(masked_array, fill_value=np.nan)
    
    return masked_array


dem_array = mask_array_with_nan(dem.data.squeeze().copy(), dem.nodata)
ref_dem_array = mask_array_with_nan(ref_dem.data.squeeze().copy(), ref_dem.nodata)

diff_array = ref_dem_array - dem_array

x_coordinates, y_coordinates = np.meshgrid(
    np.arange(diff_array.shape[1]),
    np.arange(diff_array.shape[0])
)
ramp = du.coreg.deramping(diff_array, x_coordinates, y_coordinates, 2)

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(diff_array, cmap='RdBu')
fig.colorbar(im,extend='both')
ax.set_axis_off();

# +
mask_array = np.zeros_like(diff_array)
mask_array += ramp(x_coordinates, y_coordinates)

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(
    mask_array, 
    cmap='RdBu',
#     clim=(-10, 10)
)
fig.colorbar(im,extend='both')
ax.set_title('estimated correction surface')
ax.set_axis_off();
# -

# ## Correct masked DEM

dem_array_corrected = dem_array.copy()
dem_array_corrected += ramp(x_coordinates, y_coordinates)

diff_array_corrected = ref_dem_array - dem_array_corrected

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(diff_array_corrected, cmap='RdBu')
fig.colorbar(im,extend='both')
ax.set_axis_off();

fig,ax = plt.subplots(1,2,figsize=(20,20))
im0 = ax[0].imshow(diff_array, cmap='RdBu')
# fig.colorbar(im,extend='both')
# ax[0].set_title('NMAD before: '+ f"{nmad_before:.3f} m")
# ax[0].set_axis_off();
im1 = ax[1].imshow(diff_array_corrected, cmap='RdBu')
# fig.colorbar(im1,extend='both')
# ax[1].set_title('NMAD after: '+ f"{nmad_after:.3f} m")
ax[1].set_axis_off();

dem_array_corrected.shape, ref_dem.data.shape

dem_array_corrected_raster = ref_dem.copy(new_array = dem_array_corrected)
diff_array_corrected_raster = ref_dem.copy(new_array = diff_array_corrected)

dem_array_corrected_raster.save(
    '/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/corrected.tif'
)
diff_array_corrected_raster.save(
    '/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/diff_corrected.tif'
)

# !dem_align.py \
#     {ref_dem.filename} \
#     '/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/corrected.tif'


