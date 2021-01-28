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
import fiona
import rasterio
from rasterio.mask import mask

# data_dir = '/data2/elilouis/googledrive/hsfm-geomorph/data/'
data_dir = '/data2/elilouis/hsfm-geomorph/data/'
def get_resource(file_path):
    return os.path.join(data_dir, file_path)


# ## Prepare reference DEM
# We start with a 2011 DSM downloaded from WA DNR LIDAR portal. Convert from some wierd WA state plane to the more standard UTM10N.

raw_dem = get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11.tif')
translated_dem = get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated.tif')
translated_meters_dem = get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated_meters.tif')

# !gdalwarp -t_srs epsg:32610 \
#     {raw_dem} \
#     {translated_dem}


# Convert vertical units of feet to meters

# !gdal_calc.py -A \
#     {translated_dem} \
#     --outfile={translated_meters_dem} \
#     --calc="A*0.3048"

# ### Prepare a base-surface reference DEM
# Take a road polygons and mask the reference DEM for our final alignment step.

translated_meters_masked_dem = get_resource(
    'reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated_meters_bare_surface.tif'
)

bare_roads_shp = get_resource("oly/BareRoads_WGS84/BareRoads_WGS84.shp")
bare_roads_shp_translated = get_resource("oly/BareRoads_WGS84/BareRoads_WGS84_translated.shp")

# !ogr2ogr -t_srs epsg:32610 -f "ESRI Shapefile" {bare_roads_shp_translated} {bare_roads_shp}

# +
with fiona.open(bare_roads_shp_translated) as shp:
    shapes = [feature["geometry"] for feature in shp]
    
with rasterio.open(translated_meters_dem) as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
    out_meta = src.meta
    
out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

with rasterio.open(translated_meters_masked_dem, "w", **out_meta) as dest:
    dest.write(out_image)
# -

# ## Align DEMs
# ### Use the HSFM routine

# Use first the `translated_meters_dem` for the alignment routine.
#
# Then use the `translated_meters_masked_dem` (covers bare roads only) for a single final alignment step.

input_dem = get_resource('oly/oly_dem.tif')

input_dem

translated_meters_dem

translated_meters_masked_dem

aligned_dem, transform = hsfm.asp.pc_align_p2p_sp2p(
    input_dem_file = input_dem,
    reference_dem_file = translated_meters_dem,
    output_directory = get_resource('oly/'),
    prefix     = 'run',
    p2p_max_displacement = 2000,
    sp2p_max_displacement = 1000,
    m_sp2p_max_displacement = 10,
    print_call = False,
    verbose    = False
)

# ### Use a single pc_align call with the bare surface/roads masked DEM
#
# IE align the result of the last alignment step with the DEM that JUST has the bare roads
#
#
# Is the last masked step going to cause trouble here? If so, should i do just a pc_align call? Need to figure out the details of that!

aligned_dem_file, transform = hsfm.asp.pc_align(aligned_dem,
                                                translated_meters_masked_dem,
                                                get_resource('oly/aligned_with_roads'),
                                                '--save-transformed-source-points',
                                                '--max-displacement',
                                                '25',
                                                '--alignment-method', 
                                                'similarity-point-to-point',
                                                print_call=False,
                                                verbose=True)

# ## Examine the results

aligned_dem = get_resource('oly/pc_align/run-run-run-trans_source-DEM.tif')

aligned_road_dem = get_resource('oly/aligned_with_roads/pc_align/run-trans_source-DEM.tif')

reference_dem = translated_meters_dem

# Substracts second from first

diff = hsfm.utils.difference_dems(aligned_dem, reference_dem)

hsfm.plot.plot_dem_difference_from_file_name(diff)

diff_road = hsfm.utils.difference_dems(aligned_road_dem, reference_dem)

hsfm.plot.plot_dem_difference_from_file_name(diff_road)


