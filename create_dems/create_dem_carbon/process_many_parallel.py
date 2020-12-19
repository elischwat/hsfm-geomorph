# import concurrent.futures
import math
import hsfm
import pandas as pd
import os
import matplotlib.image as mpimg
from os.path import join
from datetime import datetime
import Metashape
import logging 

# license_path = 'create_dems/create_dem_carbon/uw_agisoft.lic'
# hsfm.metashape.authentication(license_path)
# print("Checking Metadata activation...")
# print(Metashape.app.activated)

data_dir = '/home/elilouis/hsfm-geomorph/data'
input_data_path = join(data_dir, 'create_dem_carbon', 'output_data')
output_data_path = 'processing_outputs'

def process(group):
    print(f'Processing images for date {group}')
    
    project_name               = f'carbon_{group}'
    images_path                = join(input_data_path, group, 'preprocessed_images')
    images_metadata_file       = join(input_data_path, group, 'metashape_metadata.csv')
    reference_dem              = f'{data_dir}/reference_dem_highres/reference_dem_final-adj.tif'
    output_path                = join(output_data_path, group)
    focal_length               = 152 # we should really provide the actual focal length
    pixel_pitch                = 0.02
    output_dem_resolution      = 1.0 # could automate this choice in the future
    image_matching_accuracy    = 1
    densecloud_quality         = 1
    verbose                    = False
    
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

def main():
    groups = os.listdir(input_data_path)
    print(f'Processing groups: {str(groups)}')
    for date in groups:
        group_start_time = datetime.now()
        print(f'Processing {date}. Start time: {str(group_start_time)}')
        
        try:
            process(date)
            end_time =  datetime.now()
            elapsed_time = end_time - group_start_time
            print(f'Finished processing {date}. End time: {str(end_time)}. Elapsed processing time: {str(elapsed_time)}')
        except ValueError as err:
            print(f'ValueError caused failure: {err}')
            end_time =  datetime.now()
            elapsed_time = end_time - group_start_time
            print(f'Processing {date} terminated due to failure. End time: {str(end_time)}. Elapsed processing time: {str(elapsed_time)}')
        except FileExistsError as err:
            print(f'FileExistsError caused failure: {err}')
            end_time =  datetime.now()
            elapsed_time = end_time - group_start_time
            print(f'Processing {date} terminated due to failure. End time: {str(end_time)}. Elapsed processing time: {str(elapsed_time)}')
        except IndexError as err:
            print(f'IndexError caused failure: {err}')
            end_time =  datetime.now()
            elapsed_time = end_time - group_start_time
            print(f'Processing {date} terminated due to failure. End time: {str(end_time)}. Elapsed processing time: {str(elapsed_time)}')
    
if __name__ == '__main__':
    main()
