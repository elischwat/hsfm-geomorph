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

import hipp
import pandas as pd
import os

data_dir = '/data2/elilouis/hsfm-geomorph/data/oly/'
def get_resource(file_path):
    return os.path.join(data_dir, file_path)


# Create fiducial templates if you don't have them

# + jupyter={"outputs_hidden": true}
hipp.core.create_midside_fiducial_proxies_template(
    get_resource('7-4-81_OL-81_10-33-156.tif'),
    output_directory=get_resource('fiducials')
)
# -

pd.read_csv(os.path.join(data_dir, 'Oly_WS_TestPhotoApproxCenter.csv'))

hipp.batch.preprocess_with_fiducial_proxies(
    image_directory = data_dir, 
    template_directory = os.path.join(data_dir, 'fiducials'),
    output_directory = os.path.join(data_dir, 'preprocessed_images'),
)
