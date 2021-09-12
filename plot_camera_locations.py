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

import geopandas as gpd
import pandas as pd
import os
from os.path import join
# import contextily as ctx
import altair as alt

year = '91_9_9'
roll = '91V3'
base_path = (
    "/data2/elilouis/rainier_carbon_automated_timesift/"
)
sift_base_path = (
    "/data2/elilouis/rainier_carbon_automated_timesift/multi_epoch_cloud/"
)
indiv_base_path = (
    f"/data2/elilouis/rainier_carbon_automated_timesift/individual_clouds/{year}"
)


def get_gdf_from_camera_metadata(file_path):
    gdf = gpd.read_file(file_path)
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(x=gdf.lon, y=gdf.lat, z=gdf.alt, crs='EPSG:4326')
    )
    return gdf[['image_file_name', 'lon', 'lat', 'alt']]


def filter_df_for_roll(df, year_str):
    return df[df.image_file_name.str.contains(year_str)]


# +
sift_og           = filter_df_for_roll(get_gdf_from_camera_metadata(join(base_path, "metashape_metadata.csv")), roll)
sift_bundled      = filter_df_for_roll(get_gdf_from_camera_metadata(join(sift_base_path, "metaflow_bundle_adj_metadata.csv")), roll)
sift_aligned      = filter_df_for_roll(get_gdf_from_camera_metadata(join(sift_base_path, "aligned_bundle_adj_metadata.csv")), roll)


single_1973_og_0      = get_gdf_from_camera_metadata(join(indiv_base_path, "metashape_metadata.csv"))
single_1973_bundled_0 = get_gdf_from_camera_metadata(join(indiv_base_path, '0', "metaflow_bundle_adj_metadata.csv"))
single_1973_aligned_0 = get_gdf_from_camera_metadata(join(indiv_base_path, '0', "aligned_bundle_adj_metadata.csv"))

single_1973_og_1      = single_1973_aligned_0.copy()
single_1973_bundled_1 = get_gdf_from_camera_metadata(join(indiv_base_path, '1', "metaflow_bundle_adj_metadata.csv"))
single_1973_aligned_1 = get_gdf_from_camera_metadata(join(indiv_base_path, '1', "aligned_bundle_adj_metadata.csv"))

single_1973_og_2      = single_1973_aligned_1.copy()
single_1973_bundled_2 = get_gdf_from_camera_metadata(join(indiv_base_path, '2', "metaflow_bundle_adj_metadata.csv"))
single_1973_aligned_2 = get_gdf_from_camera_metadata(join(indiv_base_path, '2', "aligned_bundle_adj_metadata.csv"))

sift_og['step'] = "timesift_og"
sift_bundled['step'] = "timesift_bundled"
sift_aligned['step'] = "timesift_aligned"

single_1973_og_0['step'] = "single_1973_og_0"
single_1973_bundled_0['step'] = "single_1973_bundled_0"
single_1973_aligned_0['step'] = "single_1973_aligned_0"

single_1973_og_1['step'] = "single_1973_og_1"
single_1973_bundled_1['step'] = "single_1973_bundled_1"
single_1973_aligned_1['step'] = "single_1973_aligned_1"

single_1973_og_2['step'] = "single_1973_og_2"
single_1973_bundled_2['step'] = "single_1973_bundled_2"
single_1973_aligned_2['step'] = "single_1973_aligned_2"

cameras_sift_steps = pd.concat([
    sift_og,
    sift_bundled,
    sift_aligned
])
cameras_single_steps_0 = pd.concat([
    single_1973_og_0,
    single_1973_bundled_0,
    single_1973_aligned_0,
])
cameras_single_steps_1 = pd.concat([
    single_1973_og_1,
    single_1973_bundled_1,
    single_1973_aligned_1,
])
cameras_single_steps_2 = pd.concat([
    single_1973_og_2,
    single_1973_bundled_2,
    single_1973_aligned_2,
])


# -

def plot_camera_positions(src, title):
    cameras_plotted_north_south = alt.Chart(src).mark_circle(size=75).encode(
        x = alt.X('lat:Q', scale=alt.Scale(zero=False)),
        y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
        color = 'step:N'
    ).properties(
        title = title,width=300,height=175
    ) 
    cameras_plotted_east_west = alt.Chart(src).mark_circle(size=75).encode(
        x = alt.X('lon:Q', scale=alt.Scale(zero=False)),
        y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
        color = 'step:N'
    ).properties(
        title = title,width=300,height=175
    )
    return (cameras_plotted_north_south | cameras_plotted_east_west)


plot_camera_positions(cameras_sift_steps, "Camera positions throughout timesift")

# + jupyter={"source_hidden": true}
plot_camera_positions(cameras_single_steps_0, "Camera positions after HSFM, Iteration 0")

# + jupyter={"source_hidden": true}
plot_camera_positions(cameras_single_steps_1, "Camera positions after HSFM, Iteration 1")

# + jupyter={"source_hidden": true}
plot_camera_positions(cameras_single_steps_2, "Camera positions after HSFM, Iteration 2")
# -

start_1['step'] = 'start_1'
bundle_adj_1['step'] = 'bundle_adj_1'
align_1['step'] = 'align_1'
start_2['step'] = 'start_2'
bundle_adj_2['step'] = 'bundle_adj_2'
align_2['step'] = 'align_2'
start_3['step'] = 'start_3'
# bundle_adj_3['step'] = bundle_adj_3
# align_3['step'] = align_3
cameras_all_steps = pd.concat([
    start_1,
#     bundle_adj_1,
#     align_1,
#     start_2,
#     bundle_adj_2,
#     align_2,
#     start_3,
#     bundle_adj_3,
#     align_3,
])

cameras_plotted_north_south = alt.Chart(cameras_all_steps).mark_circle().encode(
    x = alt.X('lat:Q', scale=alt.Scale(zero=False)),
    y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
    color = 'step'
).properties(
    title = 'Camera positions at different steps',width=600,height=500
) 
cameras_plotted_east_west = alt.Chart(cameras_all_steps).mark_circle().encode(
    x = alt.X('lon:Q', scale=alt.Scale(zero=False)),
    y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
    color = 'step'
).properties(
    title = 'Camera positions at different steps',width=600,height=500
)
(cameras_plotted_north_south | cameras_plotted_east_west)

cameras_all_steps.step.unique()

# +
start_1['step'] = 'start_1'
aligned_1['step'] = 'aligned_1'
start_2['step'] = 'start_2'
aligned_2['step'] = 'aligned_2'
start_3['step'] = 'start_3'

cameras_all_steps = pd.concat([
    start_1,
    aligned_1,
    start_2,
    aligned_2,
    start_3
])
# -

cameras_plotted_north_south = alt.Chart(cameras_all_steps).mark_circle().encode(
    x = alt.X('lat:Q', scale=alt.Scale(zero=False)),
    y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
    color = 'step'
).properties(
    title = 'Camera positions at different steps'
) 
cameras_plotted_east_west = alt.Chart(cameras_all_steps).mark_circle().encode(
    x = alt.X('lon:Q', scale=alt.Scale(zero=False)),
    y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
    color = 'step'
).properties(
    title = 'Camera positions at different steps'
)
cameras_plotted_north_south | cameras_plotted_east_west

bundle_adj_cameras = os.path.join(base_path, "metaflow_bundle_adj_metadata.csv")
aligned_cameras = os.path.join(base_path, "aligned_bundle_adj_metadata.csv")
nuth_aligned_cameras = os.path.join(base_path, "nuth_aligned_bundle_adj_metadata.csv")

bundle_adj_df = get_gdf_from_camera_metadata(bundle_adj_cameras)
bundle_adj_df['step'] = '1. bundle adjusted'
aligned_df = get_gdf_from_camera_metadata(aligned_cameras)
aligned_df['step'] = '2. aligned'
nuth_df = gdf_3 = get_gdf_from_camera_metadata(nuth_aligned_cameras)
nuth_df['step'] = '3. nuth aligned'

cameras_all_steps = pd.concat([nuth_df, aligned_df, bundle_adj_df])

cameras_plotted_north_south = alt.Chart(cameras_all_steps).mark_circle().encode(
    x = alt.X('lat:Q', scale=alt.Scale(zero=False)),
    y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
    color = 'step'
).properties(
    title = 'Camera positions at different steps'
) 
cameras_plotted_east_west = alt.Chart(cameras_all_steps).mark_circle().encode(
    x = alt.X('lon:Q', scale=alt.Scale(zero=False)),
    y = alt.Y('alt:Q', scale=alt.Scale(zero=False)),
    color = 'step'
).properties(
    title = 'Camera positions at different steps'
)
cameras_plotted_north_south | cameras_plotted_east_west

bundle_adj_df = bundle_adj_df.to_crs(epsg=3857)
aligned_df = nuth_df.to_crs(epsg=3857)
nuth_df = nuth_df.to_crs(epsg=3857)

ax = bundle_adj_df.plot(figsize=(10, 10), alpha=1, color='red', edgecolor='k')
aligned_df.plot(ax=ax, figsize=(10, 10), alpha=1, color='blue', edgecolor='k')
nuth_df.plot(ax=ax, figsize=(10, 10), alpha=1, color='green', edgecolor='k')
l_lim, r_lim = ax.get_xlim()
ax.set_xlim(l_lim-1000, r_lim+1000)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery, zoom=14)

import hsfm

x_offset, y_offset, z_offset = hsfm.core.compute_point_offsets(
    "/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/73_0_0/1/metaflow_bundle_adj_metadata.csv", 
    "/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/73_0_0/2/metaflow_bundle_adj_metadata.csv"
        )
ba_CE90, ba_LE90 = (
    hsfm.geospatial.CE90(x_offset, y_offset),
    hsfm.geospatial.LE90(z_offset),
)
hsfm.plot.plot_offsets(
    ba_LE90,
    ba_CE90,
    x_offset,
    y_offset,
    z_offset,
)

dem73_fn = "/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/73_0_0/2/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-2.56_y+6.33_z+0.97_align.tif"
dem79_fn = "/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/79_10_06/2/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-0.67_y+1.14_z+0.10_align.tif"
dem87_fn = "/data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/87_08_21/2/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x-6.63_y+21.02_z+6.17_align.tif"


diff = hsfm.utils.difference_dems(dem73_fn, dem79_fn)
diff2 = hsfm.utils.difference_dems(dem79_fn, dem87_fn)

# +
import rasterio
from rasterio.plot import show_hist
from rasterio.plot import show

raster = rasterio.open(diff)
show_hist(raster, bins=50)
raster = rasterio.open(diff2)
show_hist(raster, bins=50)
# -

hsfm.plot.plot_dem_difference_from_file_name(diff, cmap='PuOr')

hsfm.plot.plot_dem_difference_from_file_name(diff2, cmap='PuOr')


