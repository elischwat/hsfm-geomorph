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

# ##### Follow instructions for installation at github.com/glaciohack/xdem

import xdem as du
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import altair as alt
import pandas as pd
import io   
from profiling_tools import *

# # Deramping, DEM Error Surface Correction

# +
dem_1973_fn = ("/data2/elilouis/rainier_friedrich/73V3/unknown/unknown/sfm/cluster_004/metashape/" +
    "sub_cluster1/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/" +
    "run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.71_y-1.04_z+0.09_align.tif"
    )
dem_1973_masked_fn = ("/data2/elilouis/rainier_friedrich/73V3/unknown/unknown/sfm/cluster_004/metashape/" +
    "sub_cluster1/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/" +
    "run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.71_y-1.04_z+0.09_align_filt.tif"
    )
 
dem_1979_fn = ("/data2/elilouis/rainier_friedrich/79V5/10/05/sfm/cluster_000/metashape0/" +
                      "pc_align/run-run-run-trans_source-DEM_dem_align/" + 
                      "run-run-run-trans_source-DEM_reference_dem_clip_nuth_x+0.15_y+0.44_z+0.19_align.tif"
                     )

dem_1979_masked_fn = ("/data2/elilouis/rainier_friedrich/79V5/10/05/sfm/cluster_000/metashape0/" +
                      "pc_align/run-run-run-trans_source-DEM_dem_align/" + 
                      "run-run-run-trans_source-DEM_reference_dem_clip_nuth_x+0.15_y+0.44_z+0.19_align_filt.tif"
                     )
dem_reference_fn = ("/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif")
# -

dem_1973           = gu.georaster.Raster(dem_1973_fn)
dem_1973_masked    = gu.georaster.Raster(dem_1973_masked_fn)
dem_1979           = gu.georaster.Raster(dem_1979_fn)
dem_1979_masked    = gu.georaster.Raster(dem_1979_masked_fn)
dem_reference = gu.georaster.Raster(dem_reference_fn)

# ## Crop to common extent (using 1973 extent)

dem_reference.crop(dem_1973)
dem_1979.crop(dem_1973)
dem_1979_masked.crop(dem_1973)

# ## Reproject to common dimensions

dem_1973_masked = dem_1973_masked.reproject(dem_1973, nodata= dem_1973_masked.nodata)
dem_reference = dem_reference.reproject(dem_1973, nodata= dem_reference.nodata)
dem_1979 = dem_1979.reproject(dem_1973, nodata=dem_1979.nodata)
dem_1979_masked = dem_1979_masked.reproject(dem_1973, nodata=dem_1979_masked.nodata)

dem_reference.crs, dem_1973_masked.crs, dem_1973.crs, dem_1979_masked.crs, dem_1979.crs, 

# ## Look at DEMs

fig, axes = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,5))
dem_1973.show(ax=axes[0], add_cb=False)
axes[0].set_title("1973 NAGAP")
dem_1973_masked.show(ax=axes[1], add_cb=False)
axes[1].set_title("1973 NAGAP, Masked")
dem_reference.show(ax=axes[2], add_cb=False)
axes[2].set_title("2007/08 LIDAR")
plt.suptitle("1973 DEMs")
plt.tight_layout()

fig, axes = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,5))
# fig, axes = plt.subplots(1,3, )
dem_1979.show(ax=axes[0], add_cb=False)
axes[0].set_title("1979 NAGAP")
dem_1979_masked.show(ax=axes[1], add_cb=False)
axes[1].set_title("1979 NAGAP, Masked")
dem_reference.show(ax=axes[2], add_cb=False)
axes[2].set_title("2007/08 LIDAR")
plt.suptitle("1979 DEMs")
plt.tight_layout()


# ## Get arrays, mask the nodata

def mask_array_with_nan(array,nodata_value):
    """
    Replace dem nodata values with np.nan.
    """
    mask = (array == nodata_value)
    masked_array = np.ma.masked_array(array, mask=mask)
    masked_array = np.ma.filled(masked_array, fill_value=np.nan)
    
    return masked_array


dem_1973_array        = mask_array_with_nan(dem_1973.data.squeeze().copy(), dem_1973.nodata)
dem_1973_masked_array = mask_array_with_nan(dem_1973_masked.data.squeeze().copy(), dem_1973_masked.nodata)
dem_reference_array   = mask_array_with_nan(dem_reference.data.squeeze().copy(), dem_reference.nodata)
dem_1979_array        = mask_array_with_nan(dem_1979.data.squeeze().copy(), dem_1979.nodata)
dem_1979_masked_array = mask_array_with_nan(dem_1979_masked.data.squeeze().copy(), dem_1979_masked.nodata)

print(dem_1973_array.shape)
print(dem_1973_masked_array.shape)
print(dem_reference_array.shape)
print(dem_1979_array.shape)
print(dem_1979_masked_array.shape)

# ## Find initial difference

diff_1973_array = dem_reference_array - dem_1973_masked_array
diff_1979_array = dem_reference_array - dem_1979_array

nmad_before = du.coreg.calculate_nmad(diff_1973_array)
print(nmad_before)

nmad_before = du.coreg.calculate_nmad(diff_1979_array)
print(nmad_before)

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(diff_1973_array, cmap='RdBu',clim=(-3, 3))
fig.colorbar(im,extend='both')
# ax.set_title(pathlib.Path(dem_fn).stem+'_elev_diff_m')
ax.set_axis_off();

# ## Fit surface

x_coordinates, y_coordinates = np.meshgrid(
    np.arange(diff_1973_array.shape[1]),
    np.arange(diff_1973_array.shape[0])
)
ramp = du.coreg.deramping(diff_1973_array, x_coordinates, y_coordinates, 2)

# +
mask_array = np.zeros_like(diff_1973_array)
mask_array += ramp(x_coordinates, y_coordinates)

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(mask_array, cmap='RdBu',clim=(-10, 10))
fig.colorbar(im,extend='both')
ax.set_title('estimated correction surface')
ax.set_axis_off();
# -

# ## Correct masked DEM

dem_masked_array_corrected = dem_masked_array.copy()
dem_masked_array_corrected += ramp(x_coordinates, y_coordinates)

elevation_diff_array_corrected = dem_reference_array - dem_masked_array_corrected

nmad_after = du.coreg.calculate_nmad(elevation_diff_array_corrected)
print(nmad_after)

fig,ax = plt.subplots(1,2,figsize=(20,20))
im0 = ax[0].imshow(elevation_diff_array, cmap='RdBu',clim=(-2, 2))
# fig.colorbar(im,extend='both')
ax[0].set_title('NMAD before: '+ f"{nmad_before:.3f} m")
ax[0].set_axis_off();
im1 = ax[1].imshow(elevation_diff_array_corrected, cmap='RdBu',clim=(-2, 2))
# fig.colorbar(im1,extend='both')
ax[1].set_title('NMAD after: '+ f"{nmad_after:.3f} m")
ax[1].set_axis_off();

# ## Apply correction to unmasked DEM

# ### First look at original difference

elevation_diff_array = dem_reference_array - dem_array

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(elevation_diff_array, cmap='RdBu',clim=(-10, 10))
fig.colorbar(im,extend='both')

# ### Then correct it

dem_array += ramp(x_coordinates, y_coordinates)

elevation_diff_array = dem_reference_array - dem_array

fig,ax = plt.subplots(figsize=(10,10))
im = ax.imshow(elevation_diff_array, cmap='RdBu',clim=(-10, 10))
fig.colorbar(im,extend='both')

# ## Write corrected DEM to disk

# should probably change the no data value back and assign more efficient dtype
corrected_dem = gu.georaster.Raster.from_array(
    data=dem_array,
    transform=dem.transform,
    crs=dem.crs,
    nodata=np.nan
)
out_fn = (
    "/data2/elilouis/rainier_friedrich/73V3/unknown/unknown/sfm/cluster_004/" +
    "metashape/sub_cluster1/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/" +
    "run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.71_y-1.04_z+0.09_align_corrected.tif"
)
corrected_dem.save(out_fn)

# ls "/data2/elilouis/rainier_friedrich/73V3/unknown/unknown/sfm/cluster_004/metashape/sub_cluster1/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/"

# # Unvertainty analysis, Road to Paradise profile

n_points = 1000
rasters_crs = 'EPSG:32610'
profiles_shapefile = "/data2/elilouis/hsfm-geomorph/data/profiles/paradise_road_xsection.shp"
files_data = """
area,            date,    filename
rainier,     NAGAP 9/24/1973,  /data2/elilouis/rainier_friedrich/73V3/unknown/unknown/sfm/cluster_004/metashape/sub_cluster1/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.71_y-1.04_z+0.09_align_corrected.tif
rainier,     NAGAP 9/24/1973 corrected,  /data2/elilouis/rainier_friedrich/73V3/unknown/unknown/sfm/cluster_004/metashape/sub_cluster1/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.71_y-1.04_z+0.09_align.tif
rainier,     USGS LIDAR 2007/08,    /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/rainier_lidar_dsm-adj.tif
"""

dem_files_df = pd.read_csv(io.StringIO(files_data), skipinitialspace=True)
dem_files_df.head(3)

gdf = gpd.read_file(profiles_shapefile)
gdf = gdf.to_crs(rasters_crs)
gdf.head(3)

gdf.geometry = gdf.geometry.apply(
    lambda line: 
    increase_line_complexity(line, n_points = n_points)
)

gdf.plot()

dem_files_df['geometry'] = gdf.geometry.iloc[0]

profile_and_dem_df = gpd.GeoDataFrame(dem_files_df)

profile_and_dem_df

profile_df_list = []
for i, row in profile_and_dem_df.iterrows():
    profile_df = get_raster_values_for_line(row.geometry, row.filename)
    profile_df['area'] = row['area']
    profile_df['date'] = row['date']
    profile_df_list.append(profile_df)
profile_df = pd.concat(profile_df_list)

profile_df

alt.Chart(profile_df).mark_line().encode(
    x = alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    y = alt.Y('raster_value:Q', title='Elevation (m)', scale=alt.Scale(zero=False)),
    color='date:N'
).properties(
    height=400, width=800,
    title={
        'text':['Uncertainty analysis, Paradise road Mt. Rainier'],
        'subtitle':['Comparing HSFM DEMs with and without error surface fitting correction']
    }
).resolve_scale(
    x='independent',
    y='independent'
)

profile_df.date.unique()

# +
src = profile_df[profile_df['date'] == 'USGS LIDAR 2007/08']
df_1973 = pd.DataFrame({
    'X': src['X'],
    'Y': src['Y'],
    'path_distance': src['path_distance'],
    'difference': profile_df[profile_df['date'] == 'USGS LIDAR 2007/08']['raster_value'] - profile_df[profile_df['date'] == 'NAGAP 9/24/1973']['raster_value'],
})
df_1973['name'] = '2007/08 - 1973'

df_1973_corrected = pd.DataFrame({
    'X': src['X'],
    'Y': src['Y'],
    'path_distance': src['path_distance'],
    'difference': profile_df[profile_df['date'] == 'USGS LIDAR 2007/08']['raster_value'] - profile_df[profile_df['date'] == 'NAGAP 9/24/1973 corrected']['raster_value']    
})
df_1973_corrected['name'] = '2007/08 - 1973 corrected'
df = pd.concat([df_1973, df_1973_corrected])
# -

df

alt.Chart(df).mark_line().encode(
    x = alt.X('path_distance:Q', title='Pathwise Distance (m)'),
    y = alt.Y('difference:Q', title='Elevation Difference (m)', scale=alt.Scale(zero=False)),
    color=alt.Color('name:N', title='DEM of Difference')
).properties(
    height=400, width=800,
    title={
        'text':['Uncertainty analysis, Paradise road Mt. Rainier'],
        'subtitle':['Comparing HSFM DEMs with and without error surface fitting correction']
    }
).resolve_scale(
    x='independent',
    y='independent'
)


