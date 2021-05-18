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

# # Mt Hood

# ### Create Mt Hood DEM from source data

# !find "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/" -name "*w001001.adf" | grep Highest_Hit | grep "2009_OLC_Hood to Coast" >> hood_relevant_tile_filepaths_adf.txt

# cat hood_relevant_tile_filepaths_adf.txt

# mkdir "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/relevant_adf_tiles/"

# !bash hood_convert_tiles_to_geotiff.sh

# ls "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/relevant_adf_tiles/"

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/

# !dem_mosaic -o /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged.tif /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/relevant_adf_tiles/*.tif


# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/

# + jupyter={"outputs_hidden": true}
# !gdal_calc.py --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=IF_SAFER --NoDataValue=-9999 --calc 'A*0.3048' -A \
#     --quiet \
#     /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged.tif \
#     --outfile /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_m.tif
# -

# !gdalwarp -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_SAFER -dstnodata -9999 -r cubic \
#     -t_srs EPSG:32610 \
#     /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_m.tif \
#     /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_m_utm.tif

# !dem_geoid --reverse-adjustment /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_m_utm.tif

# mv 2009_merged_m_utm-adj.tif /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009.tif

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/

# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged.tif
# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_m.tif
# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_merged_m_utm.tif
# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/*.txt

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/

# ### Check the Hood data by comparing with Copernicus

hood_reference_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009.tif"
hood_copernicus_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem/copernicus/Copernicus_DSM_COG_10_N45_00_W122_00_utm.tif"

hood_reference_dem_cropped = hood_reference_dem.replace(".tif", "_cropped.tif")
hood_copernicus_dem_cropped = hood_copernicus_dem.replace(".tif", "_cropped.tif")

# !gdal_translate \
#     -projwin -121.7467 45.4722  -121.6138 45.3015 \
#     -projwin_srs 'EPSG:4326' \
#     {hood_reference_dem} \
#     {hood_reference_dem_cropped}

# !gdal_translate \
#     -projwin -121.7467 45.4722  -121.6138 45.3015 \
#     -projwin_srs 'EPSG:4326' \
#     {hood_copernicus_dem} \
#     {hood_copernicus_dem_cropped}

hood_diff = hsfm.utils.difference_dems(hood_reference_dem_cropped, hood_copernicus_dem_cropped)

hood_diff

hsfm.plot.plot_dem_from_file(hood_reference_dem_cropped)

hsfm.plot.plot_dem_from_file(hood_copernicus_dem_cropped)

hsfm.plot.plot_dem_difference_from_file_name(hood_diff, spread=60)

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/

# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_cropped-diff.tif
# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/2009_cropped.tif

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem/copernicus/

# rm $hood_copernicus_dem_cropped

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem/copernicus/


