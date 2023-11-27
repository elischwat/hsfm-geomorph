#! /bin/bash

# Download these files form the WA DNR LIDAR portal. These two tiles (47 and 48) cover the peak of Mt Adams.eatge
#     mount_adams_2016_dsm_47.tif
#     mount_adams_2016_dsm_48.tif
cd /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/raw/
fn_list=$(ls *.tif)
parallel "gdal_calc.py --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=IF_SAFER --NoDataValue=-9999 --calc 'A*0.3048' -A {} --outfile {.}_m.tif" ::: $fn_list
parallel "gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -dstnodata -9999 -r cubic -s_srs ESRI:102749 -t_srs EPSG:26910 {} {.}_utm.tif" ::: *_m.tif
parallel "dem_geoid --reverse-adjustment {}" ::: *_m_utm.tif
parallel "gdal_edit.py -a_srs EPSG:32610 {}" ::: *_m_utm-adj.tif
dem_mosaic *_m_utm-adj.tif -o adams_hsfm_reference_dem.tif

#Make a low resolution version
mv adams_hsfm_reference_dem.tif ../2016.tif

gdalwarp -tr 10 10 -r cubic \
    ../2016.tif \
    ../2016_10m.tif
    
rm *_m*
