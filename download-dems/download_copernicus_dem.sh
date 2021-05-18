# Order of dem_geoid and gdalwarp matters!
north=48
west=122

raw_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM.tif"
prefix="s3://copernicus-dem-30m/Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM/${raw_dem_file}"
adj_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_DEM-adj.tif"
final_dem_file="Copernicus_DSM_COG_10_N${north}_00_W${west}_00_utm.tif"
aws s3 cp $prefix --no-sign-request ./
dem_geoid --reverse-adjustment $raw_dem_file
gdalwarp -r cubic -t_srs EPSG:32610 $adj_dem_file $final_dem_file
rm $adj_dem_file
rm $raw_dem_file