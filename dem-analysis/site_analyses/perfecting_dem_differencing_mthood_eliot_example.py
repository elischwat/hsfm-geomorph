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
import os
import rioxarray as rix

align_pairs =[
    ('1967-09', '1975-09'),
    ('1975-09', '1977-10'),
    ('1977-10', '1980-10'),
    ('1980-10', '1990-09'),
    ('1990-09', '2009')
]

diff_fns = []

for (old_dem, new_dem) in align_pairs:
    diff_fn, _ = hsfm.utils.dem_align_custom(
        "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/"  + old_dem + '.tif',
        "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/"  + new_dem + '.tif',
        "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded_aligned",
        mode='nuth',
        max_offset = 100
    )
    new_fn = "/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded_aligned/" + new_dem + '-' + old_dem + '.tif'
    diff_fns.append(new_fn)
    print(f"Saving new file to {new_fn}")
    os.rename(diff_fn, new_fn)



[rix]

import rioxarray as rix

og_raster = rix.open_rasterio("/data2/elilouis/hsfm-geomorph/data/mt_hood_eliot_glacier/rasters_regridded/1967-09.tif")
diff_raster = rix.open_rasterio(diff_fn)


