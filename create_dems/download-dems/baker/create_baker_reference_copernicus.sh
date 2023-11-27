#! /bin/bash
# DOWNLOAD TILES
north=48
west=122
raw_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM.tif"
prefix="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM/${raw_dem_file}"
aws s3 cp $prefix --no-sign-request ./


north=48
west=123
raw_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM.tif"
prefix="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM/${raw_dem_file}"
aws s3 cp $prefix --no-sign-request ./


north=49
west=122
raw_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM.tif"
prefix="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM/${raw_dem_file}"
aws s3 cp $prefix --no-sign-request ./

north=49
west=123
raw_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM.tif"
prefix="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM/${raw_dem_file}"
aws s3 cp $prefix --no-sign-request ./

original_whole_dem="baker_copernicus_reference_original.tif"

# MERGE ALL ORIGINAL TILES
dem_mosaic Copernicus_DSM_COG*.tif -o $original_whole_dem


# REVERSE ADJUST THE GEOID AND WARP TO A UTM 10N CRS
# dem_geoid must come before gdalwarp! order matters!
# Order of dem_geoid and gdalwarp matters!

adj_dem_file="baker_copernicus_reference_original-adj.tif"
dem_geoid --reverse-adjustment $original_whole_dem

final_dem_file="baker_copernicus_reference_dem.tif"
gdalwarp -r cubic -t_srs EPSG:32610 $adj_dem_file $final_dem_file







# for north in {48, 49}; do
#     for west in {122, 123}; do
#         raw_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM.tif"

#         prefix="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM/${raw_dem_file}"

#         adj_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM-adj.tif"

#         final_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_utm.tif"

#         aws s3 cp $prefix --no-sign-request ./

#         dem_geoid --reverse-adjustment $raw_dem_file

#         gdalwarp -r cubic -t_srs EPSG:32610 $adj_dem_file $final_dem_file
#     done
# done