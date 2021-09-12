import glob
import os
import rioxarray as rix
from rasterio import features
import geopandas as gpd
from shapely import geometry
from shapely.ops import unary_union

# # Find all DEMs generated

final_dems = set(glob.glob(
    "/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/**/**/cluster*/1/**/**/*align.tif", 
    recursive=True
))

dems_dict = {}
for d in os.listdir("/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/"):
    final_dems = set(glob.glob(
        f"/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/{d}/**/cluster*/1/**/**/*align.tif", 
        recursive=True
    ))
    dems_dict[d] = list(final_dems)
    print(len(final_dems))

dems_dict.keys()

dems_dict['67_9']


# # Polygonize 1 DEM

def polygonize_valid_data(file, label='', target_resolution=30, buffer_distance=100):
    """
    Params:
    file: raster file to polygonize
    target_resolution: resolution used to polygonize raster data. should be low so that not too many features are generated. defaults to 30
    buffer_distance: distance to buffer features after creation in an effort to decrease the number of individual pieces. 
                     larger buffer distance means the output geometries will approximate DEM coverage more poorly, but fewer
                     geometries are generated, which may be preferred
    """
    raster = rix.open_rasterio(file)
    raster = raster.rio.reproject(raster.rio.crs, resolution=target_resolution)
    raster = raster.where(raster == raster.rio.nodata, other=1)
    shapes_generator = features.shapes(
        raster.values[0],
        transform = raster.rio.transform()
    )
    shapes_gdf = gpd.GeoDataFrame({'json': list(shapes_generator)})
    shapes_gdf['value'] = shapes_gdf['json'].apply(lambda x: x[1])
    shapes_gdf['json'] = shapes_gdf['json'].apply(lambda x: x[0])
    shapes_gdf.geometry = shapes_gdf['json'].apply(geometry.shape)
    shapes_gdf = shapes_gdf.set_crs(raster.rio.crs)
    shapes_gdf = shapes_gdf[shapes_gdf.value == 1]
    shapes_gdf = shapes_gdf[['geometry']]
    outline = gpd.GeoDataFrame(
        geometry = [
            unary_union(
            shapes_gdf.geometry.apply(lambda x: x.buffer(buffer_distance))
        )]
    )
    outline['source_file'] = file
    outline['label'] = label
    return outline


gdf = polygonize_valid_data(
    "/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/92_09/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.24_y-0.85_z-0.21_align.tif"
)

gdf

gdf.plot()

# # Polygonize All DEMs

gdf = gpd.GeoDataFrame()
for date_key,dem_list in dems_dict.items():
    outlines = [
        polygonize_valid_data(file, label=date_key)
        for file in dem_list
    ]
    gdf = gdf.append(outlines)

gdf.plot(column='label', facecolor="none", edgecolor="black", figsize=(10,10), legend=True, linewidth=3)

gdf = gdf.set_crs(epsg=32610)
gdf.to_file(
    "/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/dem_coverage_outlines.geojson",
    driver='GeoJSON'
)

# ls /data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/

gpd.GeoDataFrame(
    geometry = gpd.GeoSeries(unary_union(gdf.geometry)),
    crs='epsg:32610'
).to_file(
    "/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/dem_coverage_max_outline.geojson",
    driver='GeoJSON'
)


