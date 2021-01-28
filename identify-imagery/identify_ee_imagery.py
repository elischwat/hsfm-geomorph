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

import hsfm
import hipp
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
import matplotlib.pyplot as plt
import contextily as ctx
import math
from itertools import chain
pd.options.display.max_columns = 100

# rainier inputs
im_df = gpd.read_file("/data2/elilouis/EE_rainier.csv")
xmin = -1.36e7
xmax = -1.349e7
ymin = 5.88e6
ymax = 5.96e6
name = "rainier"

# baker inputs
im_df = gpd.read_file("/data2/elilouis/EE_baker.csv")
# xmin = -13580000
# xmax = -13520000
# ymin = 6210000
# ymax = 6260000
xmin = -13600000
xmax = -13500000
ymin = 6190000
ymax = 6280000
name = "baker"


def get_geometry_from_corners(series):
    return shapely.wkt.loads(
        'POLYGON ((' + 
        ', '.join([
            ' '.join([series.NWlon, series.NWlat]),
            ' '.join([series.NElon, series.NElat]),
            ' '.join([series.SElon, series.SElat]),
            ' '.join([series.SWlon, series.SWlat]),
            ' '.join([series.NWlon, series.NWlat])
        ]) + '))'
    )
im_df.geometry = im_df.apply(get_geometry_from_corners, axis=1)

im_df['point_geometry'] = gpd.points_from_xy(im_df.centerLon, im_df.centerLat)

im_df = im_df.set_crs(epsg=4326)
im_df = im_df.to_crs(epsg=3857)

acquisition_date_by_photo_count = list(im_df.acquisitionDate.value_counts().keys())


def plot_date_by_project(date, df, ax):
    src = im_df[im_df.acquisitionDate == date]
    src.plot(figsize=(7.5, 7.5), alpha=0.5, edgecolor='k', column='project', legend=True, ax = ax)
    ax.set_title(f'Earth Explorer Images, {date}')


acquisition_date_by_photo_count[0], len(acquisition_date_by_photo_count)

# Set some bounds on each plot, a large square around rainier. These bounds are from the HSFM library for Rainier and are converted to EPSG 3857 (as these polygons were above)

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
            im_df,
            ax
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ctx.add_basemap(ax)
    except:
        continue
plt.tight_layout()
plt.show()
# -

fig.savefig(f"{name}_ee_images_by_project_and_date.png", dpi=300) 
