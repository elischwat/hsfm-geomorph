# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# # Prepare Reference DEMS 

import hsfm

# ## Prepare High Res Reference DEM
#
# We start with a WA DNR dataset from 2007.

dem_highres_file = '2007_final.tif'
dem_highres_file_meters = 'input_data/reference_dem_highres/reference_dem_m.tiff'
dem_highres_file_meters_warped = 'input_data/reference_dem_highres/reference_dem_m_epsg32610.tiff'
dem_highres_file_final_prefix = 'input_data/reference_dem_highres/reference_dem_final'
dem_highres_file_final = dem_highres_file_final_prefix + '-adj.tif'

hsfm.plot.plot_dem_from_file(dem_highres_file)

# ### Convert from feet to meters using gdal_calc

# !gdal_calc.py --co COMPRESS=LZW --co TILED=YES --co BIGTIFF=IF_SAFER --NoDataValue=-9999 --calc 'A*0.3048' -A $reference_dem_high_res_file --outfile $reference_dem_high_res_file_in_meters

# !gdalinfo $reference_dem_high_res_file_in_meters | head -5 | tail -1

hsfm.plot.plot_dem_from_file(reference_dem_high_res_file_in_meters)

# ### Convert from 26710 (NAD27 UTM 10N) -> 32610 (WGS84 UTM 10N) using gdalwarp
#
# ### **LAST I TRIED THIS, I HAD TO DO IT THROUGH QGIS TO GET IT WORK...**

# !gdalwarp -t_srs EPSG:32610 -r near -of GTiff $reference_dem_high_res_file_in_meters $reference_dem_high_res_file_in_meters_warped

hsfm.plot.plot_dem_from_file(reference_dem_high_res_file_in_meters_warped)

# ### Adjust the geoid

# !dem_geoid  --reverse-adjustment $reference_dem_high_res_file_in_meters_warped -o $dem_highres_file_final_prefix

# + jupyter={"outputs_hidden": true}
hsfm.plot.plot_dem_from_file(dem_highres_file_final)
# -

# ## Preare Low Res SRTM Reference DEM

# Get corner coordinates from a high resolution LIDAR DEM that is cropped to Mt. Rainier so I can download a coarse SRTM reference DEM

# !gdalinfo $dem_highres_file_final | grep "Corner Coord" --after 5

# Convert using this tool https://www.rapidtables.com/convert/number/degrees-minutes-seconds-to-degrees.html

# ## Download coarse SRTM reference DEM

LLLON = -121.948
LLLAT = 46.74643
URLON = -121.6145
URLAT = 47.00322

reference_dem = hsfm.utils.download_srtm(LLLON,
                                         LLLAT,
                                         URLON,
                                         URLAT,
                                         output_directory='input_data/reference_dem/',
                                         verbose=False)

hsfm.plot.plot_dem_from_file(reference_dem)

