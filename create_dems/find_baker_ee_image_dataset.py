import hipp

import shapely
from shapely import wkt
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import math
from itertools import chain
import os

# # Retrieve API key and set bounds for searching EE archives

apiKey = hipp.dataquery.EE_login(input(), input())

# Bounds around Mt. Baker
xmin = -122
xmax = -121
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

df = gpd.GeoDataFrame(df)

df


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
    src.plot(figsize=(7.5, 7.5), alpha=0.5, edgecolor='k', column='project', legend=True, ax = ax)
    ax.set_title(f'Earth Explorer Images, {date}')


# # Visualize extents of image sets

# +
fig, axes = plt.subplots(
    math.ceil(len(acquisition_date_by_photo_count)/5),
    5,
    sharex=True,
    sharey=True,
    figsize=(30,40)
)

axes_flat = list(chain.from_iterable(axes))

for ax, date in zip(axes_flat, acquisition_date_by_photo_count):
    try:
        plot_date_by_project(
            date, 
            df,
            ax
        )
        ax.set_xlim(xmin - 0.2, xmax + 0.2)
        ax.set_ylim(ymin - 0.2, ymax + 0.2)
#         ax.set_xlim(xmin, xmax)
#         ax.set_ylim(ymin, ymax)
#         ctx.add_basemap(ax, zoom=14, crs=df.crs, source=ctx.providers.OpenStreetMap.Mapnik)
        ctx.add_basemap(ax, crs=df.crs, source=ctx.providers.OpenStreetMap.Mapnik)
    except:
        continue
plt.tight_layout()
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

import contextily as ctx

ax = explore_df.plot(facecolor="none", figsize=(25,25), edgecolor='k')
ctx.add_basemap(ax, crs='epsg:4326')
ax.set_aspect('equal')

explore_df.to_csv('baker_ee_image_dataset.csv', index=False)


