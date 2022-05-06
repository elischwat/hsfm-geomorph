import hsfm
import xdem as du
import geoutils as gu
import numpy as np
import os
import matplotlib.pyplot as plt

import glob


base_paths = [
    "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4/47_9.0_14.0/cluster0/0/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/",
    "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4/50_9.0_2.0/cluster0/0/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/",
    "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4/79_9.0_14.0/cluster0/0/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/"
]

dem_reference_fn = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015_10m.tif"

def mask_array_with_nan(array,nodata_value):
    """
    Replace dem nodata values with np.nan.
    """
    mask = (array == nodata_value)
    masked_array = np.ma.masked_array(array, mask=mask)
    masked_array = np.ma.filled(masked_array, fill_value=np.nan)
    return masked_array

for b in base_paths:

    # GET FILE PATHS of ORIGINAL FILES
    dem_fn =        glob.glob(os.path.join(b, "*align.tif"), recursive=True)[0]
    diff_fn =       glob.glob(os.path.join(b, "*align_diff.tif"), recursive=True)[0]
    dem_masked_fn = glob.glob(os.path.join(b, "*align_filt.tif"), recursive=True)[0]
    
    print(dem_fn)
    print(diff_fn)
    print(dem_masked_fn)
    print()

    print('READING FILES')
    # OPEN FILES AS gu.georaster.Rasters
    dem = gu.georaster.Raster(dem_fn)
    ref_dem = gu.georaster.Raster(dem_reference_fn)

    print('REPROJECTING')
    dem = dem.reproject(ref_dem)
    print('CROPPING')
    dem.crop(ref_dem)

    dem_array = mask_array_with_nan(dem.data.squeeze().copy(), dem.nodata)
    ref_dem_array = mask_array_with_nan(ref_dem.data.squeeze().copy(), ref_dem.nodata)

    print('SUBTRACTING')
    diff_array = ref_dem_array - dem_array


    print('DERAMPING')
    x_coordinates, y_coordinates = np.meshgrid(
        np.arange(diff_array.shape[1]),
        np.arange(diff_array.shape[0])
    )
    ramp = du.coreg.deramping(diff_array, x_coordinates, y_coordinates, 2)

    # PLOT DIFFERENCE
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    print('PLOTTING PLOT 1/4')
    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(diff_array, cmap='RdYlBu', vmin=-2, vmax=2)
    fig.colorbar(im,extend='both')
    ax.set_axis_off()
    plt.savefig(b.replace('dem_align/', 'dem_align/plot_difference_low_modes.png'))

    print('PLOTTING PLOT 2/4')
    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(diff_array, cmap='RdYlBu', vmin=-15, vmax=15)
    fig.colorbar(im,extend='both')
    ax.set_axis_off()
    plt.savefig(b.replace('dem_align/', 'dem_align/plot_difference_high_modes.png'))



    # +
    mask_array = np.zeros_like(diff_array)
    mask_array += ramp(x_coordinates, y_coordinates)
 
    print('PLOTTING PLOT 3/4')
    fig,ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(
        mask_array, 
        cmap='RdYlBu',
    #     clim=(-10, 10)
    )
    fig.colorbar(im,extend='both')
    ax.set_title('estimated correction surface')
    ax.set_axis_off()
    plt.savefig(b.replace('dem_align/', 'dem_align/plot_estimated_correction_surface.png'))
    # -

    # ## CORRECT MASKED DEM
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    dem_array_corrected = dem_array.copy()
    dem_array_corrected += ramp(x_coordinates, y_coordinates)

    diff_array_corrected = ref_dem_array - dem_array_corrected

    fig,ax = plt.subplots(figsize=(20,20))
    im = ax.imshow(diff_array_corrected, cmap='RdYlBu', vmin=-15, vmax=15)
    fig.colorbar(im,extend='both')
    ax.set_axis_off()
    plt.savefig(b.replace('dem_align/', 'dem_align/plot_corrected_difference.png'))


    print('PLOTTING PLOT 4/4')
    # PLOT COMPARISON OF ORIGINAL AND CORRECTED DIFFERENCE
    ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
    fig,ax = plt.subplots(1,2,figsize=(20,20))
    im0 = ax[0].imshow(diff_array, cmap='RdYlBu', vmin=-15, vmax=15)
    # fig.colorbar(im,extend='both')
    # ax[0].set_title('NMAD before: '+ f"{nmad_before:.3f} m")
    # ax[0].set_axis_off();
    im1 = ax[1].imshow(diff_array_corrected, cmap='RdYlBu', vmin=-15, vmax=15)
    # fig.colorbar(im1,extend='both')
    # ax[1].set_title('NMAD after: '+ f"{nmad_after:.3f} m")
    # ax[1].set_axis_off();
    plt.savefig(b.replace('dem_align/', 'dem_align/plot_comparison.png'))

    dem_array_corrected.shape, ref_dem.data.shape

    # dem_array_corrected_raster = ref_dem.copy(new_array = dem_array_corrected)
    # diff_array_corrected_raster = ref_dem.copy(new_array = diff_array_corrected)

    # dem_array_corrected_raster.save(
    #     '/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/corrected.tif'
    # )
    # diff_array_corrected_raster.save(
    #     '/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds_backup/50_9.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/diff_corrected.tif'
    # )