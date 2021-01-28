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

# # Create DEMs of difference using the Baker mosaic results

import hsfm
import os
import geopandas as gpd

# Difference Each Baker DEM

# +
baker_dem_files = !find /data2/elilouis/baker_friedrich/ -type f -name "*DEM.tif"

baker_reference_dem_file = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif"
baker_dem_files = baker_dem_files + [baker_reference_dem_file]
# -

baker_dem_files

diff_70_77 = hsfm.utils.difference_dems('/data2/elilouis/baker_friedrich/1970-09-29_DEM.tif', '/data2/elilouis/baker_friedrich/1977-09-27_DEM.tif')

# + jupyter={"outputs_hidden": true}
diff_70_77 = hsfm.utils.difference_dems('/data2/elilouis/baker_friedrich/1970-09-29_DEM.tif', '/data2/elilouis/baker_friedrich/1977-09-27_DEM.tif')
diff_77_79 = hsfm.utils.difference_dems('/data2/elilouis/baker_friedrich/1977-09-27_DEM.tif', '/data2/elilouis/baker_friedrich/1979-10-06_DEM.tif')
diff_79_90 = hsfm.utils.difference_dems('/data2/elilouis/baker_friedrich/1979-10-06_DEM.tif', '/data2/elilouis/baker_friedrich/1990-09-05_DEM.tif')
diff_90_91 = hsfm.utils.difference_dems('/data2/elilouis/baker_friedrich/1990-09-05_DEM.tif', '/data2/elilouis/baker_friedrich/1991-09-09_DEM.tif')
diff_91_15 = hsfm.utils.difference_dems('/data2/elilouis/baker_friedrich/1991-09-09_DEM.tif',  '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/baker_2015_utm_m.tif')
# -

gdf = gpd.read_file("/data2/elilouis/hsfm-geomorph/data/dem_analysis_sediment_mask/layer.shp")
gdf = gdf.to_crs(epsg=32610)

shapes = [gdf.geometry.iloc[2]]

diffs=[
    diff_70_77,
    diff_77_79,
    diff_79_90,
    diff_90_91,
    diff_91_15
]
print(diffs)

for diff in diffs:
    hsfm.plot.plot_dem_difference_from_file_name(diff)

# Mask with `shapes`

import rasterio
from rasterio import mask
for file in diffs:
    with rasterio.open(file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(file.replace(".tif", "_clipped.tif"), "w", **out_meta) as dest:
        dest.write(out_image)

# Some older stuff...



dem_dir = "/data2/elilouis/rainier_carbon_timesift_example_dems/"
dem_file_paths = []
for file in os.listdir(dem_dir):
     dem_file_paths.append(os.path.abspath(file))

dem_file_paths

hsfm.utils.difference_dems(reference_dem_path, dem_73_cluster0, verbose=True)

hsfm.plot.plot_dem_from_file(reference_dem_path)
hsfm.plot.plot_dem_from_file(dem_73_cluster0)
hsfm.plot.plot_dem_from_file(dem_73_cluster1)
hsfm.plot.plot_dem_from_file(dem_73_cluster3)

hsfm.utils.difference_dems(reference_dem_path, dem_73_cluster0, verbose=True)

hsfm.plot.plot_dem_difference_from_file_name(
    '/home/elilouis/hsfm-geomorph/data/reference_dem_highres/reference_dem_final-adj-diff.tif')

hsfm.plot.plot_dem_difference_from_file_name(
    '/home/elilouis/hsfm-geomorph/data/hsfm_metaflow_output/73V3/cluster_000/pc_align/run-run-trans_source-DEM_dem_align/run-run-trans_source-DEM_reference_dem_clip_nuth_x-0.37_y+0.08_z+0.43_align_diff.tif',
    spread=10
)

# rm /home/elilouis/hsfm-geomorph/data/reference_dem_highres/reference_dem_final-adj-diff.tif
