# -*- coding: utf-8 -*-
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

data_dir = os.environ['hsfm_geomorph_data']

data_dir

# ## Open up KML Files
# Make sure to open all layers explicitly

# +
file_paths = ['NAGAP_1970s.kml', 'NAGAP_1980s.kml', 'NAGAP_1990s.kml']

df_list = []
for path in file_paths:
    path = os.path.join(data_dir, path)
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

df[df['roll'] == '74V5']

# ## Read in the AOIs
#
# Lets change crs to web mercator right off the bat too.

aoi_gdf = gpd.read_file(data_dir + 'aois.geojson')
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
aoi_frames_and_paths['datetime'] = pd.to_datetime(aoi_frames_and_paths.date, errors='coerce')
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

# !ls $data_dir

wau_gdf = gpd.read_file(f'{data_dir}/Watershed_Administrative_Units-shp/wau.shp')

wau_gdf.plot()

wau_in_aois_gdf = gpd.sjoin(wau_gdf, aoi_gdf)
rainier_waus_gdf = wau_in_aois_gdf[wau_in_aois_gdf.name == 'Mt. Rainier']

ax = rainier_waus_gdf.plot(column='WAU_ALIAS_',legend=True, markersize=80, legend_kwds={'bbox_to_anchor': (1.6, 1)}, alpha=0.6)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(5,5)


# ## Look at image locations and watershed delineations

# +
def plot_frames_and_aoi_polygon(points, aoi_polygon = None, lims = None):
    ax = points.plot(column='date', categorical=True, markersize=20, legend=True)
    if aoi_polygon is not None:
        gpd.GeoDataFrame(ax = ax, geometry = pd.Series(aoi_polygon)).plot(legend_kwds={'bbox_to_anchor': (1.6, 1)}, edgecolor='red', lw=2, facecolor="none")
    ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
    if lims is not None:
        ax.set(xlim=lims[0], ylim=lims[1])
    plt.gcf().set_size_inches(8,8)
    
import math 
def plot_frames_and_aoi_date_separated(points, aoi_polygon = None, lims=None):
    groupby = points.groupby('date')
    fig, axes = plt.subplots(math.ceil(len(groupby.size().tolist())/4),4, figsize=(20,20), sharex=True, sharey=True)
    axes_flat = [item for sublist in axes for item in sublist]
    for key, group in groupby:
        ax = axes_flat.pop(0)
        if aoi_polygon is not None:
            gpd.GeoDataFrame(geometry = pd.Series(aoi_polygon)).plot(ax=ax, legend_kwds={'bbox_to_anchor': (1.6, 1)}, edgecolor='red', lw=2, facecolor="none")
        group.plot(ax=ax, column='date', categorical=True, markersize=40, legend=True)
        ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain)
    while len(axes_flat) > 0:
        ax = axes_flat.pop(0)
        if aoi_polygon is not None:
            gpd.GeoDataFrame(geometry = pd.Series(aoi_polygon)).plot(ax=ax, legend_kwds={'bbox_to_anchor': (1.6, 1)}, edgecolor='red', lw=2, facecolor="none")
        group.plot(ax=ax, column='date', categorical=True, markersize=40, legend=True)
        ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain)
#     plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)


# -

ax = rainier_waus_gdf.plot(legend_kwds={'bbox_to_anchor': (1.6, 1)}, edgecolor='red', lw=2, facecolor="none")
ax = rainier_frames_gdf.plot(column='date', categorical=True, markersize=20, ax=ax)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
ax.set(xlim=(-1.357e7,-1.3535e7), ylim=(5.905e6, 5.94e6))
plt.gcf().set_size_inches(20,20)

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
x, y, arrow_length = 0.5, 0.5, 0.1
ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
            arrowprops=dict(facecolor='black', width=5, headwidth=15),
            ha='center', va='center', fontsize=20,
            xycoords=ax.transAxes)

ax = fryingpan_frames_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldTopoMap)
plt.gcf().set_size_inches(14,14)

plot_frames_and_aoi_polygon(fryingpan_frames_df, None)

plot_frames_and_aoi_date_separated(fryingpan_frames_df, None)

# ## Look at data in smaller watersheds, Nisqually and Carbon

rainier_sub_aois = gpd.read_file(f"{data_dir}/rainier_sub_aois.geojson")
rainier_sub_aois = rainier_sub_aois.to_crs(epsg=3857)
nisqually_polygon = rainier_sub_aois[rainier_sub_aois.name=='nisqually'].geometry.iloc[0]
carbon_polygon = rainier_sub_aois[rainier_sub_aois.name=='carbon'].geometry.iloc[0]
nisqually_frames = rainier_frames_gdf[rainier_frames_gdf.geometry.within(nisqually_polygon)]
carbon_frames = rainier_frames_gdf[rainier_frames_gdf.geometry.within(carbon_polygon)]

len(nisqually_frames), len(carbon_frames)

plot_frames_and_aoi_polygon(nisqually_frames, nisqually_polygon)

plot_frames_and_aoi_date_separated(nisqually_frames, nisqually_polygon)

plot_frames_and_aoi_polygon(carbon_frames, carbon_polygon)

plot_frames_and_aoi_date_separated(carbon_frames, carbon_polygon)


# # Save Datasets to CSV for the HSFM Pipeline

def create_targets_list(kml_derived_df, output_path):
    #Open image name/UUID dataset
    pids_df = pd.read_csv(f'{data_dir}/glacier_names_pids.csv')
    filenames = kml_derived_df.apply(lambda r: ('NAGAP_' + r.roll + '_' + r.frame).replace(' ', ''), axis=1)
    pid_df = pids_df[pids_df.fileName.isin(filenames)]
    pid_df[[
        'Year','Date','Location','Latitude','Longitude','Altitude','fileName','pid_tn','pid_jpeg','pid_tiff','_merge'
    ]].to_csv(output_path, index=None)
    return output_path


pids_df = pd.read_csv(f'{data_dir}/glacier_names_pids.csv')

# ### Nisqually 1977

src = nisqually_frames.groupby('date').get_group('1977-02-11')
print(len(src))
create_targets_list(
    src,
    'targets_nisqually_1977.csv'
)

# ### Nisqually 1980

src = nisqually_frames.groupby('date').get_group('1980-9-10')
print(len(src))
create_targets_list(
    src,
    'targets_nisqually_1980.csv'
)

# ### Nisqually All

# +
src = nisqually_frames
print(len(src))
create_targets_list(
    src,
    'targets_nisqually_all_dates.csv'

)
# -

# ### Carbon All

# +
src = carbon_frames
print(len(src))
create_targets_list(
    src,
    'targets_carbon_all_dates.csv'

)
# -
# ### Frying Pan Watershed All

# +
src = fryingpan_frames_df
print(len(src))
create_targets_list(
    src,
    'targets_carbon_all_dates.csv'

)
# -

src.roll = src.roll.apply(lambda x: x.strip())

src.roll.iloc[0]

src.Name.apply(lambda x: x[:4]) == src.roll

src[src.Name.apply(lambda x: x[:4]) != src.roll]

# ### Bandaid  - Missing Lat/Long values
#
#
# I later noticed missing Lat/Long values for a subset of images in 1974. Fix that here by getting lat/long info from the KML files.
#
# Note also that "Name" and "roll" columns do not agree.

fixing = pd.read_csv('targets_carbon_all_dates.csv')

fixing[fixing.Year == 1974]

carbon_frames['fileName'] = 'NAGAP_' + carbon_frames.roll + '_' + carbon_frames.frame
to_merge = carbon_frames[carbon_frames.roll=='74V5'][['fileName', 'latitude', 'longitude']]
to_merge

fixing.loc[fixing['fileName'].str.startswith('NAGAP_74V5_')]

# These rows are in the same order so I can go ahead and assign the lat long values from the `to_merge` dataframe.

fixing.loc[fixing['fileName'].str.startswith('NAGAP_74V5_'), 'Latitude'] = to_merge['latitude'].tolist()
fixing.loc[fixing['fileName'].str.startswith('NAGAP_74V5_'), 'Longitude'] = to_merge['longitude'].tolist()

fixing.loc[fixing['fileName'].str.startswith('NAGAP_74V5_')]

fixing.to_csv('targets_carbon_all_dates.csv')


