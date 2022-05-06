import hipp
import shapely
from shapely import wkt
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from itertools import chain
import os
import rasterio
import matplotlib
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
import contextily as ctx

# # Retrieve API key and set bounds for searching EE archives

# +
from getpass import getpass
username = input()
password = getpass()

apiKey = hipp.dataquery.EE_login(username, password)
# -

# Bounds around Mt. Baker
xmin = -122
xmax = -121.5
ymax =  49
ymin = 48.5

startDate = '1901-01-01'
endDate   = '2022-01-01'

label     = 'test_download'
output_directory = '/data2/elilouis/baker-ee'

maxResults   = 50000

scenes = hipp.dataquery.EE_sceneSearch(
    apiKey,
    xmin,ymin,xmax,ymax,
    startDate,endDate,
    maxResults   = maxResults
)

df = hipp.dataquery.EE_filterSceneRecords(scenes)

# ## Remove NAGAP images

df = df[~df['project'].str.contains('NAG')]

# ## Filter out not available images

df = df[df.hi_res_available=='Y']

df = gpd.GeoDataFrame(df)


def get_geometry_from_corners(series):
    return shapely.wkt.loads(
        'POLYGON ((' + 
        ', '.join([
            ' '.join([str(series.NWlon), str(series.NWlat)]),
            ' '.join([str(series.NElon), str(series.NElat)]),
            ' '.join([str(series.SElon), str(series.SElat)]),
            ' '.join([str(series.SWlon), str(series.SWlat)]),
            ' '.join([str(series.NWlon), str(series.NWlat)])
        ]) + '))'
    )
df['geometry'] = df.apply(get_geometry_from_corners, axis=1)
df = df.set_crs(epsg=4326)

acquisition_date_by_photo_count = list(df.acquisitionDate.value_counts().keys())


def plot_date_by_project(date, df, ax):
    src = df[df.acquisitionDate == date]
    src.plot(
#         figsize=(7.5, 7.5), 
        alpha=0.5, edgecolor='k', column='project', legend=True, ax = ax)
    ax.set_title(f'{date}, {len(src)} images')


# # Visualize extents of image sets

# +
fig, axes = plt.subplots(
    math.ceil(len(acquisition_date_by_photo_count)/5),
    5,
    sharex=True,
    sharey=True,
    figsize=(30,40)
)

axes_flat = axes.ravel()

for ax, date in zip(axes_flat, acquisition_date_by_photo_count):
    try:
        plot_date_by_project(
            date, 
            df,
            ax
        )
        ax.set_xlim(xmin - 0.2, xmax + 0.2)
        ax.set_ylim(ymin - 0.2, ymax + 0.2)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
#         ctx.add_basemap(ax, crs=df.crs, 
#                         source=ctx.providers.Esri.WorldImagery
#                        )
    except:
        continue
# plt.tight_layout()
plt.show()
# -

# # Identify projects of interest

projects = [
    'LK000', 
    '18900', 
    '94000'
]

df = df[df['project'].isin(projects)]

len(df.groupby('project').get_group('LK000'))

len(df.groupby('project').get_group('18900'))

len(df.groupby('project').get_group('94000'))

# # Identify single project

df = df[df['project']=='LK000']

df.iloc[0]

# ## Use already generate image footprings to examine LK000 dataset in detail

src = gpd.read_file('/data2/elilouis/generate_ee_dems_baker/mixed_timesift/timesifted_image_footprints.geojson')
src = src[src['filename'].str.contains('LK000')]

src = src[src.filename.str[-5]=='1']
# src = src[src.filename.str[-5]=='2']

src.to_crs('EPSG:32610').to_file('LK000_footprints.geojson', driver='GeoJSON')

# !gdal_rasterize -burn 1 -tr 30 30 -ot UInt32 -a_nodata 0 -add LK000_footprints.geojson LK000_footprints.tif


ax = src.plot(facecolor="none", edgecolor='k', linewidth=0.5,figsize=(10,12))
ctx.add_basemap(ax, crs = src.crs, source=ctx.providers.Esri.DeLorme)
plt.axis('off')

# +
import matplotlib as mpl
fig = plt.gcf()
size = fig.get_size_inches()
raster_src = rasterio.open("LK000_footprints.tif")

maximus = max(raster_src.read(1).ravel())
cmap = matplotlib.cm.get_cmap(
    'rainbow', 
    maximus-1
)
cmap.set_bad('white') 
data = raster_src.read(1)

import numpy as np
data = np.ma.masked_values(data, 0)
data = np.ma.masked_values(data, 1)

plt.imshow(
    data, 
    cmap=cmap, 
#     aspect='auto', 
    )
cb = plt.colorbar()
plt.axis('off')
# -

# # Explore how many images are in this project

scenes = hipp.dataquery.EE_sceneSearch(
    apiKey,
    -122.1,48.0,-121.6,49.0, # iteratively searched and modifided these bounds to get a good batch of images surrounding baker (there is more than what i use here, but i cut it off)
    '1950-09-01','1950-09-03',
    maxResults   = maxResults
)
explore_df = hipp.dataquery.EE_filterSceneRecords(scenes)
explore_df['geometry'] = explore_df.apply(get_geometry_from_corners, axis=1)
explore_df = gpd.GeoDataFrame(explore_df, geometry='geometry').set_crs(epsg=4326)

explore_df['project'].unique()

explore_df = explore_df[explore_df['project']=='LK000']

ax = explore_df.plot(facecolor="none", figsize=(25,25), edgecolor='k')
ctx.add_basemap(ax, crs='epsg:4326')
ax.set_aspect('equal')

explore_df.to_csv('baker_ee_image_dataset.csv', index=False)


