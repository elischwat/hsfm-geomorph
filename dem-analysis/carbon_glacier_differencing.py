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

# updated run:

dem_files = !find /data2/elilouis/rainier_carbon_timesift/rainier_carbon_post_timesift_hsfm/ -type f -name "*align.tif" | grep /2/ 

diff_fn = hsfm.utils.difference_dems(dem_73_fn, dem_07_fn)

# + jupyter={"outputs_hidden": true}
diff_dem = rxr.open_rasterio(diff_fn, masked=True).squeeze()

# + jupyter={"outputs_hidden": true}
hsfm.plot.plot_dem_difference_from_file_name(diff_fn, spread=10)
# -

diff_dem.hvplot.image(
    cmap='RdBu', color=(-10,10)
).redim(
    value=dict(range=(-10, 10))
).opts(
    xaxis=None, 
    yaxis=None,
    show_frame=False,
)

# C

# ### Crop all Carbon DEMs to fluvial mask

# +
dem_73_fn = "/data2/elilouis/rainier_carbon/input_data/73V3/00/00/sfm/cluster_000/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-2.30_y+5.33_z+1.14_align.tif"
dem_79_fn = "/data2/elilouis/rainier_carbon/input_data/79V5/10/06/sfm/cluster_000/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/run-run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.73_y+1.32_z+0.23_align.tif"
dem_91_fn = "/data2/elilouis/rainier_carbon/input_data/91V3/09/09/sfm/cluster_000/metashape0/pc_align/run-run-run-trans_source-DEM_dem_align/run-run-run-trans_source-DEM_reference_dem_clip_nuth_x+1.44_y+7.60_z+3.60_align.tif"

dem_07_fn = '/home/elilouis/hsfm-geomorph/data/reference_dem_highres/reference_dem_final-adj.tif'
# -

mask = '/data2/elilouis/hsfm-geomorph/data/dem_analysis_sediment_mask/layer.shp'
crop_extent = gpd.read_file(mask)[1:]

dem_73 = rxr.open_rasterio(dem_73_fn, masked=True).squeeze()
dem_79 = rxr.open_rasterio(dem_79_fn, masked=True).squeeze()
dem_91 = rxr.open_rasterio(dem_91_fn, masked=True).squeeze()
dem_07 = rxr.open_rasterio(dem_07_fn, masked=True).squeeze()

dem_73_clipped = dem_73.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
dem_79_clipped = dem_79.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
dem_91_clipped = dem_91.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
dem_07_clipped = dem_07.rio.clip(crop_extent.geometry.apply(mapping), crop_extent.crs)
dem_73_clipped_fn = "/data2/elilouis/hsfm-geomorph/data/carbon_glacier_differencing/dem_73_clipped.tif"
dem_79_clipped_fn = "/data2/elilouis/hsfm-geomorph/data/carbon_glacier_differencing/dem_79_clipped.tif"
dem_91_clipped_fn = "/data2/elilouis/hsfm-geomorph/data/carbon_glacier_differencing/dem_91_clipped.tif"
dem_07_clipped_fn = "/data2/elilouis/hsfm-geomorph/data/carbon_glacier_differencing/dem_07_clipped.tif"

dem_73_clipped.rio.to_raster(dem_73_clipped_fn)
dem_79_clipped.rio.to_raster(dem_79_clipped_fn)
dem_91_clipped.rio.to_raster(dem_91_clipped_fn)
dem_07_clipped.rio.to_raster(dem_07_clipped_fn)

# ls /data2/elilouis/hsfm-geomorph/data/carbon_glacier_differencing/

# Plot all the clipped data

dem_73_clipped.hvplot.image( cmap='RdBu').opts(
    xaxis=None, yaxis=None,
    width=200,height=250,
    show_frame=False,
) + dem_79_clipped.hvplot.image( cmap='RdBu').opts(
    xaxis=None, yaxis=None,
    width=200,height=250,
    show_frame=False,
) + dem_91_clipped.hvplot.image( cmap='RdBu').opts(
    xaxis=None, yaxis=None,
    width=200,height=250,
    show_frame=False,
) + dem_07_clipped.hvplot.image( cmap='RdBu').opts(
    xaxis=None, yaxis=None,
    width=200,height=250,
    show_frame=False,
)

diff_73_79_fn = hsfm.utils.difference_dems(dem_79_clipped_fn, dem_73_clipped_fn)
diff_79_91_fn = hsfm.utils.difference_dems(dem_91_clipped_fn, dem_79_clipped_fn)
diff_91_07_fn = hsfm.utils.difference_dems(dem_07_clipped_fn, dem_91_clipped_fn)


src = rxr.open_rasterio(diff_73_79_fn, masked=True).squeeze()
src.hvplot.image( cmap='RdBu').redim(
    value=dict(range=(-20, 20))
).opts(
    xaxis=None, yaxis=None,
    width=400,height=500,
    show_frame=False,
)

src = rxr.open_rasterio(diff_79_91_fn, masked=True).squeeze()
src.hvplot.image( cmap='RdBu').redim(
    value=dict(range=(-20, 20))
).opts(
    xaxis=None, yaxis=None,
    width=400,height=500,
    show_frame=False,
)

src = rxr.open_rasterio(diff_91_07_fn, masked=True).squeeze()
src.hvplot.image( cmap='RdBu').redim(
    value=dict(range=(-20, 20))
).opts(
    xaxis=None, yaxis=None,
    width=400,height=500,
    show_frame=False,
)


