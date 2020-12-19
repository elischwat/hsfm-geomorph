#! /usr/bin/env python
import hsfm
import pandas as pd
import os
import matplotlib.image as mpimg
from os.path import join
from datetime import datetime
import Metashape
print(Metashape.app.activated)

data_dir = '/home/elilouis/hsfm-geomorph/data'

output_data_path = join(data_dir, 'create_dem_carbon', 'output_data')

os.listdir(output_data_path)

dates = [
    '10-6-79',
    '7-28-92',
    '8-10-74',
    '9-13-90', #<- Fails
    '10-6-92',
    '10-9-80',
    '11-2-97',
    '1362'
]

# ## Notes from Friedrich:
#
# pc_align/run-trans_source-DEM.tif <- the first DEM product, one run of alignment using point-to-plane alignment (gross translation and rotation)
# pc_align/run-run-trans_source-DEM.tif <- second DEM product, one run of alignment using similarity point-to-point alignment (fixes overall scaling)
#  see hsfm.asp.pc_align_p2p_sp2p where both of those ASP calls are made
#
# FINAL DEM PRODUCT pc_align/run-run-trans_source-DEM.tif
# FINAL with nuuth-kab dem alignment and differencing postprocessing:  pc_align/run-run-trans_source-DEM_dem_align/
#
#
#  pc_align/run-run-trans_source-DEM_dem_align
#    This does the nuth and kaab alignment, it mostly won't do much.
#
#  Might need a better bare ground masks than the NLCD, because there is a lot of vegetation in my areas of focus
#
#  Best possible DEM resolution is ~ 2.5* the Ground resolution, which can be found in the metashape report, page 2

for date in dates:
    now = datetime.now()
    print(f'Processing images for date {date}')
    
    project_name               = f'carbon_{date}'
    images_path                = join(output_data_path, date, 'preprocessed_images')
    images_metadata_file       = join(output_data_path, date, 'metashape_metadata.csv')
    reference_dem              = f'{data_dir}/reference_dem_highres/reference_dem_final-adj.tif'
    output_path                = join('processing_outputs', date)
    focal_length               = 152
    pixel_pitch                = 0.02
    output_dem_resolution      = 1.0
    image_matching_accuracy    = 1
    densecloud_quality         = 1
    verbose                    = True
    
    print('project_name:\t\t' + project_name)
    print('images_path:\t\t' + images_path)
    print('images_metadata_file:\t' + images_metadata_file)
    print('reference_dem:\t\t' + reference_dem)
    print('output_path:\t\t' + output_path)
    
    hsfm.batch.metaflow(project_name,
                        images_path,
                        images_metadata_file,
                        reference_dem,
                        output_path,
                        focal_length,
                        pixel_pitch,
                        image_matching_accuracy = image_matching_accuracy,
                        densecloud_quality      = densecloud_quality,
                        verbose                 = True,
                        cleanup                 = True)
    print("Elapsed time", str(datetime.now() - now))


