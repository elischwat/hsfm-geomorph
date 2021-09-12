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


def get_gdf_from_camera_metadata(file_path):
    gdf = gpd.read_file(file_path)
    gdf = gpd.GeoDataFrame(
        gdf, geometry=gpd.points_from_xy(x=gdf.lon, y=gdf.lat, z=gdf.alt, crs='EPSG:4326')
    )
    return gdf[['image_file_name', 'lon', 'lat', 'alt']]


cams = get_gdf_from_camera_metadata("/data2/elilouis/hood_ee_test/metashape_metadata.csv")
cams_bundle_adj = get_gdf_from_camera_metadata("/data2/elilouis/hood_ee_test/metashape_metadata_bundle_adj.csv")
cams_bundle_adj_aligned = get_gdf_from_camera_metadata("/data2/elilouis/hood_ee_test/metashape_metadata_bundle_adj_aligned.csv")


def plot_camera_positions(src, title, zero=False):
    cameras_plotted_north_south = alt.Chart(src).mark_circle(size=75).encode(
        x = alt.X('lat:Q', scale=alt.Scale(zero=zero)),
        y = alt.Y('alt:Q', scale=alt.Scale(zero=zero)),
        color = 'step:N'
    ).properties(
        title = title,width=300,height=175
    ) 
    cameras_plotted_east_west = alt.Chart(src).mark_circle(size=75).encode(
        x = alt.X('lon:Q', scale=alt.Scale(zero=zero)),
        y = alt.Y('alt:Q', scale=alt.Scale(zero=zero)),
        color = 'step:N'
    ).properties(
        title = title,width=300,height=175
    )
    cameras_plotted_lat_lon = alt.Chart(src).mark_circle(size=75).encode(
        x = alt.X('lon:Q', scale=alt.Scale(zero=zero)),
        y = alt.Y('lat:Q', scale=alt.Scale(zero=zero)),
        color = 'step:N'
    ).properties(
        title = title,width=300,height=175
    ) 
    return (cameras_plotted_north_south | cameras_plotted_east_west) | cameras_plotted_lat_lon


# # Look at camera positions after running pipeline with only Mt Hood EE images

# + jupyter={"source_hidden": true}
plot_camera_positions(cams, "Unaligned EE cameras, original metadata") 

# + jupyter={"source_hidden": true}
plot_camera_positions(cams_bundle_adj, "Bundle adjusted EE cameras, bundle adjusted with only EE") 

# + jupyter={"source_hidden": true}
plot_camera_positions(cams_bundle_adj_aligned, "PC aligned and bundle adjusted EE cameras, bundle adjusted with only EE") 
# -

# # Look at camera positions after running pipeline with only Mt Hood EE and NAGAP images mixed

# We need to extract camera positions from the project first...I didn't use the pipeline to run the mix, just a notebook `hsfm/examples/EE_example.ipynb`

cams_mixed_fn = "/data2/elilouis/hood_ee_nagap_mixed/metashape_metadata.csv"
cams_mixed = pd.read_csv(cams_mixed_fn)

cams_mixed_bundle_adj, removed_cams = hsfm.metashape.update_ba_camera_metadata(
    metashape_project_file = "/data2/elilouis/hood_ee_nagap_mixed/metashape_old/project.psx",
    metashape_metadata_csv = cams_mixed_fn
)

# Remove non-ee images

cams_mixed = cams_mixed[cams_mixed.image_file_name.str.contains('1VDYL')]

cams_mixed_bundle_adj = cams_mixed_bundle_adj[cams_mixed_bundle_adj.image_file_name.str.contains('1VDYL')]

removed_cams = removed_cams[removed_cams.image_file_name.str.contains('1VDYL')]

plot_camera_positions(cams_mixed, "Unaligned EE cameras, original metadata") 

# + jupyter={"source_hidden": true}
plot_camera_positions(cams_mixed_bundle_adj, "Bundle adjusted EE cameras, bundle adjusted with NAGAP images") 

# + jupyter={"source_hidden": true}
plot_camera_positions(removed_cams, "") 
# -

len(cams), len(cams_bundle_adj), len(cams_bundle_adj_aligned), len(cams_mixed), len(cams_mixed_bundle_adj)

footprints_mixed_gdf = hsfm.metashape.image_footprints_from_project("/data2/elilouis/hood_ee_nagap_mixed/metashape_old/project.psx")

ee_footprints_mixed_gdf = footprints_mixed_gdf[footprints_mixed_gdf.filename.str.contains('1VDYL')]

ee_footprints_mixed_gdf.to_crs(crs='epsg:32610').to_file("/data2/elilouis/hood_ee_nagap_mixed/metashape_old/image_footprints.geojson", driver='GeoJSON')

# !gdal_rasterize -burn 1 -tr 5 5 -ot UInt32 -a_nodata 0 -add /data2/elilouis/hood_ee_nagap_mixed/metashape_old/image_footprints.geojson /data2/elilouis/hood_ee_nagap_mixed/metashape_old/image_footprints.tif

import contextily as ctx
ax = ee_footprints_mixed_gdf.to_crs(epsg=3857).plot(alpha=0.3, edgecolor='k', figsize=(10,10))
ctx.add_basemap(ax)

ax = footprints_mixed_gdf.to_crs(epsg=3857).plot(alpha=0.3, edgecolor='k', figsize=(10,10))
ctx.add_basemap(ax)

good_nagap_cams = gpd.read_file('/data2/elilouis/mt_hood_timesift/timesifted_image_footprints.geojson')

ax = good_nagap_cams.to_crs(epsg=3857).plot(alpha=0.5, edgecolor='k', figsize=(10,10))
ee_footprints_mixed_gdf.to_crs(epsg=3857).plot(ax=ax, alpha=0.15, color='orange', edgecolor='k', figsize=(10,10), label='EE')
ctx.add_basemap(ax)

ax = good_nagap_cams.to_crs(epsg=3857).plot(alpha=0.5, edgecolor='k', figsize=(10,10))
ee_footprints_mixed_gdf.to_crs(epsg=3857).plot(ax=ax, alpha=0.15, color='orange', edgecolor='k', figsize=(10,10), label='EE')
ctx.add_basemap(ax)
ax.set_xlim(-1.3555E7, -1.3538E7)
ax.set_ylim(5.67E6, 5.695E6)


