#! /bin/bash
# laz files downloaded from USGS ftp server: ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MountAdams_2016_LAS_2017/
# Create directory to put laz files
mkdir /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/laz/
# Download all laz files
wget -m ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/USGS_LPC_WA_MountAdams_2016_LAS_2017/laz/ -P /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/laz/

#Retrieve first returns or only returns from LIDAR laz files
# the pdal settings are available in the json file
# Init projection is EPSG:26910, adjusted to NAVD88 Datum
# projection info was gleaned by the metadata.xml obtained by downloading any 1 las.zip from the las/ folder
#cmd
parallel --progress "pdal pipeline first_return_filter.json  --readers.las.filename={} --writers.las.filename={.}_first_return.laz" ::: /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/laz/**.laz
# Use ASP point2dem to grid the point cloud tiles
# DEM posted at 1 m resolution, with 95 percentile stats filter for aggregrating points within the grid
#cmd
parallel --progress "point2dem --t_srs EPSG:32610 --tr 1 --filter 95-pct {}" ::: *first_return.laz
# compose a VRT file from all the tiled DEMs
#cmd
gdalbuildvrt adams_lidar_dsm.vrt *DEM.tif
# convert to cloud optimised geotiff
#cmd 
gdal_translate -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER -co COPY_SRC_OVERVIEWS=YES -co COMPRESS_OVERVIEW=YES -co NUM_THREADS=ALL_CPUS -co PREDICTOR=3 adams_lidar_dsm.vrt adams_lidar_dsm.tif
# adjust for geoid
#cmd
dem_geoid --reverse-adjustment adams_lidar_dsm.tif