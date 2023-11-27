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

# ### Check the Adams data by comparing with Copernicus

reference_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/2016.tif"
copernicus_dem = "/data2/elilouis/hsfm-geomorph/data/reference_dem/copernicus/Copernicus_DSM_COG_10_N46_00_W122_00_utm.tif"

reference_dem_cropped = reference_dem.replace(".tif", "_cropped.tif")
copernicus_dem_cropped = copernicus_dem.replace(".tif", "_cropped.tif")

# !gdal_translate \
#     -projwin -121.5857 46.2708 -121.4036 46.1195 \
#     -projwin_srs 'EPSG:4326' \
#     {reference_dem} \
#     {reference_dem_cropped}

# !gdal_translate \
#     -projwin -121.5857 46.2708 -121.4036 46.1195 \
#     -projwin_srs 'EPSG:4326' \
#     {copernicus_dem} \
#     {copernicus_dem_cropped}

diff = hsfm.utils.difference_dems(reference_dem_cropped, copernicus_dem_cropped)

diff

hsfm.plot.plot_dem_from_file(reference_dem_cropped)

hsfm.plot.plot_dem_from_file(copernicus_dem_cropped)

hsfm.plot.plot_dem_difference_from_file_name(diff, spread=60)

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/

# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/2016_cropped-diff.tif
# !rm /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/2016_cropped.tif

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem_highres/adams/

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem/copernicus/

# rm $copernicus_dem_cropped

# ls /data2/elilouis/hsfm-geomorph/data/reference_dem/copernicus/


