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

import rioxarray as rix
import os
import hvplot
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

# ## Load the 1970 - 2009 DOD

file = '/data2/elilouis/field excursion - mt hood/dods/1967_2009_dod.tif'

dod = rix.open_rasterio(file, masked=True).squeeze()

# ### Distribution

sns.distplot(dod.values.flatten())

# ### View Raster

plt.figure(figsize=(10,10))
plt.imshow(dod, cmap='PuOr', vmin=-30,vmax=30)

# ## Load Erosion Polygons

erosion_polygons = gpd.read_file('/data2/elilouis/hsfm-geomorph/data/mt_hood_mass_wasted/erosion_polygons.geojson')

erosion_polygons

# ### Calculate mass wasted in each

erosion_polygons.id = erosion_polygons.index

erosion_polygons['mass_wasted_raw_dod'] = erosion_polygons.geometry.apply(
    lambda geom:
    (
        dod.rio.clip([geom]).sum()*(
            dod.rio.resolution()[0]*(-dod.rio.resolution()[1])
        )
    ).values
)

erosion_polygons

erosion_polygons.to_file('/data2/elilouis/hsfm-geomorph/data/mt_hood_mass_wasted/erosion_polygons.geojson', driver='GeoJSON')
