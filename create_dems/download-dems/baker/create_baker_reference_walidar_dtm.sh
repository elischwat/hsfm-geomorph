#! /bin/bash

# Download these files form the WA DNR LIDAR portal. We use all the tiles available to get maximum coverage.
#       baker_2015_dtm_10.tif
#       baker_2015_dtm_11.tif
#       baker_2015_dtm_12.tif
#       baker_2015_dtm_13.tif
#       baker_2015_dtm_14.tif
#       baker_2015_dtm_16.tif
#       baker_2015_dtm_17.tif
#       baker_2015_dtm_18.tif
#       baker_2015_dtm_2.tif
#       baker_2015_dtm_3.tif
#       baker_2015_dtm_6.tif
#       baker_2015_dtm_7.tif
#       baker_2015_dtm_8.tif
#       baker_2015_dtm_9.tif

cd /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/dtm
fn_list=$(ls *.tif)
parallel "gdal_calc.py --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=IF_SAFER --NoDataValue=-9999 --calc 'A*0.3048' -A {} --outfile {.}_m_dtm.tif" ::: $fn_list
parallel "gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -dstnodata -9999 -r cubic -s_srs ESRI:102749 -t_srs EPSG:26910 {} {.}_utm.tif" ::: *_m_dtm.tif
parallel "dem_geoid --reverse-adjustment {}" ::: *_m_dtm_utm.tif
parallel "gdal_edit.py -a_srs EPSG:32610 {}" ::: *_m_dtm_utm-adj.tif
dem_mosaic *_m_dtm_utm-adj.tif -o baker_hsfm_reference_dtm.tif

#Make a low resolution version
mv baker_hsfm_reference_dtm.tif ../2015_dtm.tif

gdalwarp -tr 10 10 -r cubic \
    ../2015_dtm.tif \
    ../2015_dtm_10m.tif
    
rm *_m*
