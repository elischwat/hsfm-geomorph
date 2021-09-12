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

# %load_ext autoreload
# %autoreload 2
import hipp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import glob
import hipp
import os
import numpy as np
import hsfm


project = "21200"

apiKey = hipp.dataquery.EE_login(input(), input())

# Bounds around Glacier Peak
xmin = -121.3
xmax = -120.9
ymax = 48.2
ymin = 47.9

startDate = '1901-01-01'
endDate   = '2022-01-01'

label     = 'test_download'
output_directory = '/data2/elilouis/earth_explorer_dem_gla/'

maxResults   = 50000

scenes_df = hipp.dataquery.EE_sceneSearch(
    apiKey,
    xmin,ymin,xmax,ymax,
    startDate, endDate,
    maxResults = maxResults
)
df = hipp.dataquery.EE_filterSceneRecords(scenes_df)

pd.options.display.max_columns=100

image_type_dict = {
    12: 'Black & White Infrared',
    13: 'Color Infrared',
    14: 'Color',
    24: 'Black & White'
}

df['imageType'] = df['imageType'].apply(image_type_dict.get)

df[df['hi_res_available']=='Y'].sort_values('altitudesFeet')

# +
from shapely.geometry import box

df['geometry'] = df.apply(
    lambda x: box(x['SWlon'], x['SWlat'], x['NElon'], x['NElat']),
    axis='columns'
)
# -

df.groupby('acquisitionDate').apply(lambda x: x['project'].unique())

df.groupby('acquisitionDate').apply(lambda x: x['imageType'].unique())

df.groupby('acquisitionDate').apply(len)

example_df = df.groupby('acquisitionDate').get_group('1985/08/16')

import geopandas as gpd
example_df = gpd.GeoDataFrame(example_df)
example_df = example_df.set_crs(epsg=4326)

import contextily as ctx
ax = example_df.plot(figsize=(10, 10), alpha=0.2, edgecolor='k')
ctx.add_basemap(ax, zoom=12, source=ctx.providers.OpenTopoMap, crs=example_df.crs)
ax.set_axis_off()



src = df[df['project']==project]

entityIds = list(src['entityId'])

import hipp

images_directory, calibration_reports_directory = hipp.dataquery.EE_downloadImages(
    apiKey,
    entityIds
)


