#! /usr/bin/env python
import hsfm
import pandas as pd
import os
import matplotlib.image as mpimg
from os.path import join

data_dir = '/Volumes/GoogleDrive/My Drive/hsfm-geomorph/data/'
license_path = 'uw_agisoft.lic'

dates = [
#     '10-6-79',
#     '7-28-92',
#     '8-10-74',
    '9-13-90',
#     '10-6-92',
#     '10-9-80',
#     '11-2-97',
#     '1362'
]

import Metashape
print(Metashape.app.activated)

hsfm.metashape.authentication(license_path)

for date in dates:
    now = datetime.now()
    print(f'Processing images for date {date}')
    
    project_name               = f'carbon_{date}'
    images_path                = join('output_data', date, 'preprocessed_images')
    images_metadata_file       = join('output_data', date, 'metashape_metadata.csv')
    reference_dem              = f'{data_dir}/reference_dem_highres/reference_dem_final-adj.tif'
    output_path                = join('processing_outputs', date)
    focal_length               = 152
    pixel_pitch                = 0.02
    output_dem_resolution      = 0.5
    image_matching_accuracy    = 4
    densecloud_quality         = 4
    verbose                    = True
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

'create_dems/create_dem_carbon/uw_agisoft.lic'
