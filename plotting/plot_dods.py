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
import rioxarray as rxr
import geopandas as gpd
from shapely.geometry import mapping
import seaborn as sns
import matplotlib.pyplot as plt
import hvplot

cluster_03_dem_diff = '/data2/elilouis/rainier/input_data/73V3/sfm/cluster_003/metashape2/pc_align/run-run-trans_source-DEM_dem_align/run-run-trans_source-DEM_reference_dem_clip_nuth_x-1.30_y+0.60_z-1.95_align_diff.tif'
cluster_02_dem_diff = '/data2/elilouis/rainier/input_data/73V3/sfm_modified_clusters/cluster_002/metashape3/pc_align/run-run-trans_source-DEM_dem_align/run-run-trans_source-DEM_reference_dem_clip_nuth_x-2.20_y+6.10_z+0.83_align_diff.tif'
mask = '/data2/elilouis/hsfm-geomorph/data/dem_analysis_sediment_mask/layer.shp'


import rioxarray as rxr


# +
# hsfm.plot.plot_dem_difference_from_file_name(cluster_02_dem_diff)
carbon_whole = -rxr.open_rasterio(cluster_02_dem_diff, masked=True).squeeze()

carbon_whole.hvplot.image(
    cmap='RdBu'
).redim(
    value=dict(range=(-10, 10))
).opts(
    xaxis=None, 
    yaxis=None,
    width=400,
    height=500,
    show_frame=False,

)
# -

hsfm.plot.plot_dem_difference_from_file_name(cluster_03_dem_diff)

# ## Crop 

# Differencing was done backwards... make sure to reverse here.

carbon_raster = - rxr.open_rasterio(cluster_02_dem_diff, masked=True).squeeze()
emmons_raster = - rxr.open_rasterio(cluster_03_dem_diff, masked=True).squeeze()
crop_extent = gpd.read_file(mask)

# ### Crop Emmons

emmons_clipped = emmons_raster.rio.clip(
        crop_extent.geometry.apply(mapping),# Needed to match polygon to tiff CRS
        crop_extent.crs
    )

sns.distplot(emmons_clipped.values.flatten())

emmons_clipped.hvplot.image(
    cmap='RdBu', color=(-10,10)
).redim(
    value=dict(range=(-10, 10))
).opts(
    xaxis=None, 
    yaxis=None,
    show_frame=False,
)

# ### Crop Carbon

carbon_clipped = carbon_raster.rio.clip(
        crop_extent.geometry.apply(mapping),# Needed to match polygon to tiff CRS
        crop_extent.crs
    )

sns.distplot(carbon_clipped.values.flatten())

# +
carbon_clipped.hvplot.image(
    cmap='RdBu'
).redim(
    value=dict(range=(-10, 10))
).opts(
    xaxis=None, 
    yaxis=None,
    width=400,
    height=500,
    show_frame=False,

)
# -


