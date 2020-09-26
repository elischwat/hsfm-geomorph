#! /usr/bin/env python
import hsfm
from datetime import datetime
now = datetime.now()
project_name               = 'nisqually_1980'
images_path                = 'input_data/images_preprocessed/'
images_metadata_file       = 'input_data/images_preprocessed/metashape_metadata.csv'
reference_dem = '/home/elilouis/hsfm-geomorph/input_data/reference_dem_highres/reference_dem_final-adj.tif'
output_path                = 'routine_outputs/'
focal_length               = 152
pixel_pitch                = 0.02
output_dem_resolution      = 0.5
image_matching_accuracy    = 1
densecloud_quality         = 1
metashape_licence_file     = '/home/elilouis/hsfm/uw_agisoft.lic'
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