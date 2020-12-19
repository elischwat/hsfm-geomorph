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

data_dir = '/data2/elilouis/hsfm-geomorph/data/'
def get_resource(file_path):
    return os.path.join(data_dir, file_path)

# Translate to UTM 10N 

# !gdalwarp -t_srs epsg:32610 \
#     {get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11.tif')} \
#     {get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated.tif')}


# Convert feet to meters

# + jupyter={"outputs_hidden": true}
# !gdal_calc.py -A \
#     {get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated.tif')} \
#     --outfile={get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated_meters.tif')} \
#     --calc="A*0.3048"
# -

input_dem = get_resource('oly/oly_dem4.tif')
reference_dem = get_resource('reference_dem_highres/oly/oesf_2014/dsm/oesf_2014_dsm_11_translated_meters.tif')

hsfm.asp.pc_align_p2p_sp2p(
    input_dem_file = input_dem,
    reference_dem_file = reference_dem,
    output_directory = get_resource('oly/'),
    prefix     = 'run',
    p2p_max_displacement = 2000,
    sp2p_max_displacement = 1000,
    m_sp2p_max_displacement = 10,
    print_call = False,
    verbose    = False
)

aligned_dem = get_resource('oly/pc_align/run-run-run-trans_source-DEM.tif')

# Substracts second from first

hsfm.utils.difference_dems(aligned_dem, reference_dem)

diff_dem = get_resource('oly/pc_align/run-run-run-trans_source-DEM-diff.tif')

hsfm.plot.plot_dem_difference_from_file_name(diff_dem)


