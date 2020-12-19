#! /usr/bin/env python
import hsfm
from datetime import datetime
now = datetime.now()

data_dir = '/home/elilouis/hsfm-geomorph/data'
license_path = '/home/elilouis/hsfm/uw_agisoft.lic'

project_name               = 'nisqually_1977'
images_path                = 'input_data/images_preprocessed/'
images_metadata_file       = 'input_data/metashape_metadata.csv'
reference_dem              = f'{data_dir}/reference_dem_highres/reference_dem_final-adj.tif'
output_path                = 'routine_outputs/'
focal_length               = 152
pixel_pitch                = 0.02
output_dem_resolution      = 0.5
image_matching_accuracy    = 4
densecloud_quality         = 4
metashape_licence_file     = license_path
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
                    metashape_licence_file  = metashape_licence_file,
                    verbose                 = True,
                    cleanup                 = True)
print("Elapsed time", str(datetime.now() - now))
