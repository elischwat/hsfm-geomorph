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

# # Identify Imagery

# +
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt 
import contextily as ctx

import os
import cv2
import fiona 
import geopandas as gpd
import re
import pandas as pd
# enable fiona KML driver
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
# -

# ## Open up KML Files
# Make sure to open all layers explicitly

# +
file_paths = ['NAGAP_1970s.kml', 'NAGAP_1980s.kml', 'NAGAP_1990s.kml']

df_list = []
for path in file_paths:
    for layer in fiona.listlayers(path):
        try:
            df_list.append(gpd.read_file(path, driver='KML', layer=layer))
        except ValueError:
            None
df = pd.concat(df_list)
# -

len(df)

df.head()

# Change CRS to Web Mercator for easy plotting

df = df.to_crs(epsg=3857)


# ## Parse the Description data column

# +
def parse_description(row):
        s = row.Description
        if s == '' or s is None:
            return pd.Series([None,None,None,None,None,None,None,None,None])
        else:
            lines = re.split(r'(<br>|</br>)\s*', s)
            src = next((i for i in lines if 'src' in i), None)
            date = next((i for i in lines if 'Date' in i), None)
            location = next((i for i in lines if 'Location' in i), None)
            roll = next((i for i in lines if 'Roll' in i), None)
            frame = next((i for i in lines if 'Frame' in i), None)
            latitude  = next((i for i in lines if 'Latitude' in i), None)
            longitude  = next((i for i in lines if 'Longitude' in i), None)
            altitude  = next((i for i in lines if 'Altitude' in i), None)
            type_  = next((i for i in lines if 'Type' in i), None)
            return pd.Series([
                None if src is None else src.split(':')[-1].replace('"/>', "").replace("//", ""),
                None if date is None else date.split(':')[-1],
                None if location is None else location.split(':')[-1],
                None if roll is None else roll.split(':')[-1],
                None if frame is None else frame.split(':')[-1],
                None if latitude is None else latitude.split(':')[-1].replace('°', ''),
                None if longitude is None else longitude.split(':')[-1].replace('°', ''),
                None if altitude is None else altitude.split(':')[-1],
                None if type_ is None else type_.split(':')[-1]
            ])

df[['src', 'date', 'location', 'roll', 'frame', 
    'latitude', 'longitude', 'altitude', 'type']] = df.apply(parse_description, axis=1)
# -

len(df)

df.head(3)

# ## Read in the AOIs
#
# Lets change crs to web mercator right off the bat too.

aoi_gdf = gpd.read_file('aois.geojson')
aoi_gdf = aoi_gdf.to_crs(epsg=3857)

ax = aoi_gdf.plot()
ctx.add_basemap(ax)
plt.gcf().set_size_inches(8,8)

baker_polygon = aoi_gdf[aoi_gdf.name == 'Mt. Baker'].geometry.iloc[0]
glacier_polygon = aoi_gdf[aoi_gdf.name == 'Glacier Peak'].geometry.iloc[0]
rainier_polygon = aoi_gdf[aoi_gdf.name == 'Mt. Rainier'].geometry.iloc[0]

# ## Look at locations of all images

len(df.date.unique())

src = df[df.geometry.type=='Point']
ax = src.plot(markersize=0.25, facecolor='red')
ctx.add_basemap(ax)
plt.gcf().set_size_inches(8,8)

# ## Look at locations of images in our AOIs

df.crs.to_epsg(), aoi_gdf.crs.to_epsg()

aoi_frames_and_paths = gpd.sjoin(df, aoi_gdf)

# Format date column...

# +
aoi_frames_and_paths['datetime'] = pd.to_datetime(aoi_frames_and_paths.date)
aoi_frames_and_paths.date = aoi_frames_and_paths.datetime.dt.date

aoi_frames_df = aoi_frames_and_paths[
    aoi_frames_and_paths.geometry.type=='Point']

aoi_paths_df = aoi_frames_and_paths[
    aoi_frames_and_paths.geometry.type!='Point']

# -

# Fix the data for the paths
#  
# For all the path rows, `Name` really contains the `date` and `Name` columns smushed together

aoi_paths_df['date'] = aoi_paths_df.Name.apply(lambda x: pd.to_datetime(x.split('-')[-1]))
aoi_paths_df['Name'] = aoi_paths_df.Name.apply(lambda x: x.split('-')[0])

ax = aoi_frames_df.plot(markersize=7, facecolor='red', legend=True, 
                        column='date', categorical=True, 
                        legend_kwds={'bbox_to_anchor': (1.6, 1)})
plt.gcf().set_size_inches(8,8)
ax.set(xlim=(-1.37e7,-1.345e7), ylim=(5.87e6,6.3e6))
ctx.add_basemap(ax)

ax = aoi_paths_df.plot(linewidth=1, column='date', categorical=True, legend=True,
                      legend_kwds={'bbox_to_anchor': (1.6, 1)})
plt.gcf().set_size_inches(8,8)
ax.set(xlim=(-1.37e7,-1.345e7), ylim=(5.87e6,6.3e6))
ctx.add_basemap(ax)

# ## Examine image dates 

aoi_paths_df.date.unique(),aoi_frames_df.date.unique()

set(aoi_paths_df.date.unique()).difference(set(aoi_frames_df.date.unique()))

set(aoi_frames_df.date.unique()).difference(set(aoi_paths_df.date.unique()))

# # Identify Mt Rainier Imagery

# ## Look at all images on Mt. Rainier

rainier_frames_gdf = aoi_frames_df[
    aoi_frames_df.geometry.within(rainier_polygon)]

ax = rainier_frames_gdf.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(10,10)

# ## Look at all images in watershed-delineated subsections of Mt Rainier

# If I want to focus on the Nisqally Glacier/River system, it looks like I should investigate imagery from all dates... I need a polygon for the Nisqally watershed.

# ## Load Washington watershed geometries

wau_gdf = gpd.read_file('/home/elilouis/Watershed_Administrative_Units-shp/wau.shp')

wau_gdf.plot()

wau_in_aois_gdf = gpd.sjoin(wau_gdf, aoi_gdf)
rainier_waus_gdf = wau_in_aois_gdf[wau_in_aois_gdf.name == 'Mt. Rainier']

ax = rainier_waus_gdf.plot(column='WAU_ALIAS_',legend=True, markersize=80, legend_kwds={'bbox_to_anchor': (1.6, 1)}, alpha=0.6)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(5,5)

# ## Look at image locations and watershed delineations

ax = rainier_waus_gdf.plot(legend_kwds={'bbox_to_anchor': (1.6, 1)}, edgecolor='red', lw=2, facecolor="none")
ax = rainier_frames_gdf.plot(column='date', categorical=True, markersize=20, ax=ax)
ctx.add_basemap(ax, source=ctx.providers.Stamen.TerrainBackground)
ax.set(xlim=(-1.357e7,-1.3535e7), ylim=(5.905e6, 5.94e6))
plt.gcf().set_size_inches(10,10)

# The Kautz, Carbon, and Frying Pan watersheds look to have lots of images on different dates

# ## Look at image locations in the Kautz, Carbon, and Frying Pan Watersheds

wau_gdf.WAU_ALIAS_.where(
    wau_gdf.WAU_ALIAS_.str.contains('KAUTZ', na=False)
).dropna()

wau_gdf.WAU_ALIAS_.where(
    wau_gdf.WAU_ALIAS_.str.contains('CARBON', na=False)
).dropna()

wau_gdf.WAU_ALIAS_.where(
    wau_gdf.WAU_ALIAS_.str.contains('FRYING', na=False)
).dropna()

kautz_frames_df = aoi_frames_df[aoi_frames_df.geometry.within(wau_gdf.geometry.iloc[594])]
carbon_frames_df = aoi_frames_df[aoi_frames_df.geometry.within(wau_gdf.geometry.iloc[536])]
fryingpan_frames_df = aoi_frames_df[aoi_frames_df.geometry.within(wau_gdf.geometry.iloc[564])]

ax = kautz_frames_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap)
plt.gcf().set_size_inches(14,14)

ax = carbon_frames_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap)
plt.gcf().set_size_inches(14,14)

ax = fryingpan_frames_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap)
plt.gcf().set_size_inches(14,14)

# ## Look at data in a smaller watershed, say the Nisqually

rainier_sub_aois = gpd.read_file("rainier_sub_aois.geojson")
rainier_sub_aois = rainier_sub_aois.to_crs(epsg=3857)
nisqually_polygon = rainier_sub_aois[rainier_sub_aois.name=='nisqually'].geometry.iloc[0]
carbon_polygon = rainier_sub_aois[rainier_sub_aois.name=='carbon'].geometry.iloc[0]
nisqually_frames = rainier_frames_gdf[rainier_frames_gdf.geometry.within(nisqually_polygon)]
carbon_frames = rainier_frames_gdf[rainier_frames_gdf.geometry.within(carbon_polygon)]

len(nisqually_frames), len(carbon_frames)


def plot_frames_and_aoi_polygon(points, polygon, lims = None):
    ax = gpd.GeoDataFrame(geometry = pd.Series(polygon)).plot(legend_kwds={'bbox_to_anchor': (1.6, 1)}, edgecolor='red', lw=2, facecolor="none")
    points.plot(column='date', categorical=True, markersize=20, ax=ax, legend=True)
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    if lims is not None:
        ax.set(xlim=lims[0], ylim=lims[1])
    plt.gcf().set_size_inches(8,8)


plot_frames_and_aoi_polygon(nisqually_frames, nisqually_polygon)

plot_frames_and_aoi_polygon(carbon_frames, carbon_polygon)

plot_frames_and_aoi_polygon(carbon_frames, carbon_polygon, lims = ((-1.3566e7, -1.3550e7), (5.918e6, 5.9405e6)))

# ## Visualize all images in the Nisqually river valley, separated by date

groupby_date =local_frames_gdf.groupby('date')

# +
fig, axes = plt.subplots(4,4, figsize=(20,20))
axes_flat = [item for sublist in axes for item in sublist]
for key, group in local_frames_gdf.groupby('date'):
    ax = axes_flat.pop(0)
    ax = group.plot(column='date', categorical=True, markersize=40, legend=True, ax=ax)
    ax.set(xlim =(-1.35575e7, -1.35500e7), ylim=(5.906e6, 5.913e6))
    ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain)
    
plt.show()
# -

# # Gather Data to Create a DEM - One Date in the Kautz Watershed

# ## Visaulize one-date images

# ### Nisqally Glacier Stream, 1977

DATE = '1977-02-11'

nisqually_1977_df = kautz_frames_df[kautz_frames_df.date==DATE]
ax = nisqually_1977_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(8,8)

# There we go, a single nice river valley.
#
# We can create a DEM with these.
#
# Lets find all the tiff UUIDs for these images

# ### Nisqually Glacier Stream, 1980

DATE = '1980-09-10'

nisqually_1980_df = kautz_frames_df[kautz_frames_df.date==DATE]
ax = nisqually_1980_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(8,8)

# A not nice collection from a single day.
#
# Let's test the pipeline and see what happens when we use these images. Lets find all the tiff UUIDs for these images..
#
# If it's annoying, we can try creating a dem with the following subset:

# Frame numbers for nearby images: 132, 133, 134, 136, 137, 138, 139 (note missing frame 135?)

src = nisqually_1980_df[nisqually_1980_df.frame==' 138']

nisqually_1980_subset_df = nisqually_1980_df.sort_values('frame').reset_index().iloc[:7]
ax = nisqually_1980_subset_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(8,8)

# # Functions for creating target CSV.

pids_df = pd.read_csv('glacier_names_pids.csv')


def create_targets_list(kml_derived_df, output_path):
    #Open image name/UUID dataset
    pids_df = pd.read_csv('glacier_names_pids.csv')
    filenames = kml_derived_df.apply(lambda r: ('NAGAP_' + r.roll + '_' + r.frame).replace(' ', ''), axis=1)
    pid_df = pids_df[pids_df.fileName.isin(filenames)]
    pid_df[[
        'Year','Date','Location','Latitude','Longitude','Altitude','fileName','pid_tn','pid_jpeg','pid_tiff','_merge'
    ]].to_csv(output_path, index=None)
    return output_path


create_targets_list(nisqually_1977_df, 'create_dem_nisqually_1977/targets_nisqually_1977.csv')

create_targets_list(nisqually_1980_df, 'create_dem_nisqually_1980/targets_nisqually_1980.csv')

create_targets_list(nisqually_1980_subset_df, 'create_dem_nisqually_1980/targets_nisqually_1980_subset.csv')
