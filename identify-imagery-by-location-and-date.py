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
aoi_frames_and_paths.date = pd.to_datetime(aoi_frames_and_paths.date)

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

# # Gather Data to Create a DEM - One Date in the Kautz Watershed

# ## Visaulize one-date images

kautz_frames_df = kautz_frames_df[kautz_frames_df.date=='1977-02-11']
ax = kautz_frames_df.plot(column='date', categorical=True, legend=True, markersize=80)
ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(8,8)

# There we go, a single nice river valley
#
# Lets find all the tiff UUIDs for these images

# ## Open image name/UUID dataset

pids_df = pd.read_csv('glacier_names_pids.csv')
pids_df.head()

# ### Create the `fileName` that is used in the `glacier_names_pids.csv` dataset in the Kautz watershed images datasets

kautz_watershed_1977_filenames = kautz_frames_df.apply(lambda r: 'NAGAP_' + r.roll + '_' + r.frame, axis=1)
kautz_watershed_1977_filenames

kautz_watershed_1977_pid_df = pids_df[pids_df.fileName.isin(kautz_watershed_1977_filenames)]

kautz_watershed_1977_pid_df.pid_tiff

# ## Download Thumbnails

# + jupyter={"outputs_hidden": true}
import hsfm.core

for pid in kautz_watershed_1977_pid_df.pid_tiff:
    print('Processing ' + pid + '...')
    img_gray = hsfm.core.download_image(pid)
    outdir = 'thumbnails'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out = os.path.join(outdir, pid + '.tif')
    cv2.imwrite(out,img_gray)
    final_output = hsfm.utils.optimize_geotif(out)
    os.remove(out)
    os.rename(final_output, out)
# -

# ## Show a thumbnail

# +
pil_im = Image.open('thumbnails/urn:uuid:25401745-4c2f-429d-8ee5-709149d25024.tif') #Take jpg + png

im_array = np.asarray(pil_im)
fig = plt.imshow(im_array)
fig = plt.gcf()
fig.set_size_inches(20,20)
plt.show()
# -

# # Output targets.csv file with list of images for Kautz Watershed, 1977

kautz_watershed_1977_pid_df[[
    'Year','Date','Location','Latitude','Longitude','Altitude','fileName','pid_tn','pid_jpeg','pid_tiff','_merge'
]].to_csv('create_dem_kautz_watershed_1977/targets_kautz_watershed_1977.csv', index=None)

# # Find 60s imagery (no KML file to use)

# ## Find 60s images listed in the `glacier_names_pids.csv` dataset

images_60s = pids_df[pids_df.Year < 1970]
images_60s.head()

len(images_60s), len(images_60s[~images_60s.Location.isna()]), len(images_60s[~images_60s.Longitude.isna()]), len(images_60s[~images_60s.Latitude.isna()])

# ## Find 60s images on Mt. Rainier

# ### Open the glacier polygon dataset, which will help us identify images lacking lat/long info

rgi_gdf = gpd.read_file('/home/elilouis/02_rgi60_WesternCanadaUS.shp')
rgi_gdf = rgi_gdf.to_crs(epsg=3857)

rainier_glaciers_gdf = rgi_gdf[rgi_gdf.geometry.within(rainier_polygon)]

ax = rainier_glaciers_gdf.plot(column='Name')
ctx.add_basemap(ax, url=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(8,8)

rainier_glaciers_gdf.Name.unique()

rainier_glaciers_gdf.Name = rainier_glaciers_gdf.Name.apply(lambda x: x.replace('WA','').strip())
rainier_glacier_names = rainier_glaciers_gdf.Name.unique()
rainier_glacier_names

rainier_images_60s = images_60s[images_60s.Location.isin(rainier_glacier_names)]
rainier_images_60s

# ### Plot glaciers we have images for

rainier_glaciers_with_images_gdf = rainier_glaciers_gdf[rainier_glaciers_gdf.Name.isin(rainier_images_60s.Location)]

ax = rainier_glaciers_with_images_gdf.plot(column='Name', legend=True)
ctx.add_basemap(ax, url=ctx.providers.Esri.WorldImagery)
plt.gcf().set_size_inches(5,5)

# And how many images do i have per glacier from the 60s? Not many

rainier_images_60s.Location.value_counts()

# So if we want 60s data, maybe the Carbon and Frying Pan Watersheds are preferable over Kautz.

rainier_images_60s

rainier_images_60s_gdf = gpd.GeoDataFrame(
    rainier_images_60s, geometry=gpd.points_from_xy(rainier_images_60s.Longitude, rainier_images_60s.Latitude))
rainier_images_60s_gdf.crs = {'init' :'epsg:4326'}
rainier_images_60s_gdf = rainier_images_60s_gdf.to_crs(epsg=3857)

ax = rainier_glaciers_with_images_gdf.plot(column='Name', legend=True)
rainier_images_60s_gdf.plot(ax=ax, color='red')

# It seems that these images were all from two points, so maybe we don't have many 1960s images at all for Mt Rainier. We should examine the thumbnails to be sure.
