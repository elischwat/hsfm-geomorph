import pandas as pd
import geopandas as gpd
import rioxarray as rix

dod_fn = '/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4//70_9.0_29.0/dod.tif'
dod_raster = rix.open_rasterio(dod_fn, masked=True)

valley_bounds_fn = '/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/valley_bounds.geojson'
valley_bounds_gdf = gpd.read_file(valley_bounds_fn)

valley_bounds_gdf

dod_raster.rio.reproject(
    dod_raster.rio.crs,
    resolution = (30, 30)
).plot(vmin=-20, vmax=20, 
       cmap='RdYlBu_r'
)


# !dem_mask.py -h

# !dem_mask.py --glaciers --nlcd --nlcd_filter not_forest {dod_fn}
# # !dem_mask.py --nlcd_filter --nlcd not_forest {dod_fn}

dod_masked_raster = rix.open_rasterio(
    "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4/70_9.0_29.0/dod_ref.tif",
    masked=True
)

dod_masked_raster.rio.reproject(
    dod_raster.rio.crs,
    resolution = (30, 30)
).plot(
    vmin=-20, 
    vmax=20, 
    cmap='RdYlBu_r'
)


dod_raster.rio.resolution()

from xrspatial.hillshade import hillshade

dod_raster.rio.clip(
    valley_bounds_gdf.loc[valley_bounds_gdf.name == 'Mazama'].geometry.to_list()
).plot(
    vmin=-20, 
    vmax=20, 
    cmap='RdYlBu_r'
)

len(pd.Series(dod_raster.values.flatten()).dropna())/10e6, len(pd.Series(dod_masked_raster.values.flatten()).dropna())/10e6

dod_masked_raster.rio.clip(
    valley_bounds_gdf.loc[valley_bounds_gdf.name == 'Mazama'].geometry.to_list()
).plot(
    vmin=-20, 
    vmax=20, 
    cmap='RdYlBu_r'
)

valley_bounds_gdf


