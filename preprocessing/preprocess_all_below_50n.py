#! /usr/bin/env python

import hipp

from datetime import datetime
import os
import glob
import pandas as pd

now = datetime.now()
out_dir = '/home/elilouis/hsfm-geomorph/preprocessing/outputs/'
template_parent_dir = '/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap'
nagap_image_metadata_csv_path = '/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv'

template_dirs = sorted(glob.glob(os.path.join(template_parent_dir, '*')))
template_types = []
for i in template_dirs:
    template_types.append(i.split('/')[-1])

df = pd.read_csv(nagap_image_metadata_csv_path)
df = df[df['fiducial_proxy_type'].isin(template_types)]


for i,v in enumerate(template_types):
    df_tmp = df[df['fiducial_proxy_type']  == v].copy()
    df_tmp = df_tmp.loc[df_tmp['Latitude'] < 50]
    
    image_directory = hipp.dataquery.NAGAP_download_images_to_disk(df_tmp,
                                                                   output_directory=os.path.join(out_dir, 
                                                                                                 'input_data', 
                                                                                                 v+'_raw_images'))
    template_directory = template_dirs[i]
    
    hipp.batch.preprocess_with_fiducial_proxies(image_directory,
                                                template_directory,
                                                output_directory=os.path.join(out_dir, 
                                                                              'qc',
                                                                              v+'_cropped_images'),
                                                qc_df_output_directory=os.path.join(out_dir, 
                                                                                    'qc',
                                                                                    v+'_proxy_detection_data_frames'),
                                                qc_plots_output_directory=os.path.join(out_dir, 
                                                                                       'qc',
                                                                                       v+'_proxy_detection'))

print("Elapsed time", str(datetime.now() - now))