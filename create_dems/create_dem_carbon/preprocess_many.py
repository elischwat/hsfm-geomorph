# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
import pandas as pd
import os
import matplotlib.image as mpimg

# ls /Volumes/GoogleDrive/My\ Drive/hsfm-geomorph/data/

data_dir = '/Volumes/GoogleDrive/My Drive/hsfm-geomorph/data/'
license_path = '/home/elilouis/hsfm/uw_agisoft.lic'

hsfm.metashape.authentication(license_path)

# +
image_directory = 'input_data/images_source'
output_data_dir = 'output_data'

targets_file = '../../identify-imagery/targets_carbon_all_dates.csv'

image_df = pd.read_csv(targets_file)
# -

# ## Download all tiffs

batch_number = 1
for date, data in image_df.groupby('Date'):
    data.to_csv('temp.csv', index=None)
    print(f'Downloading batch {batch_number}')
    hsfm.batch.download_images_to_disk(
        'temp.csv', 
        output_directory=os.path.join(image_directory,  str(date).replace('/','-'))
    )
    # !rm temp.csv
    batch_number = batch_number + 1


# ## What images were meant to be downloaded that were not?

def list_all_tif_files_in_dir(directory):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    ls = []
    for x in os.walk(directory):
        ls.append(flatten(list(x)[1:]))
    return [token for token in flatten(ls) if token.endswith('.tif')]


downloaded_images = list_all_tif_files_in_dir(image_directory)

len(image_df.fileName), len(downloaded_images)

set(downloaded_images).difference(set(image_df.fileName+'.tif'))

set(image_df.fileName+'.tif').difference(set(downloaded_images))

# ## Create fiducial marker templates for each date

# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

for date_dir in os.listdir(image_directory):
    fiducial_template_dir = os.path.join('fiducials/', date_dir)
    img_name = os.listdir(os.path.join(image_directory, date_dir))[0]
    img_path = os.path.join(os.path.join(image_directory, date_dir), img_name)
    print(f'Creating fiducials with image {img_path} for {date_dir} to output_directory {fiducial_template_dir}.')

    img = mpimg.imread(img_path)
    hsfm.utils.create_fiducials(img, output_directory=fiducial_template_dir)


# ls $data_dir

fiducial_dir = os.path.join(data_dir, 'fiducials/nagap')

date_fiducial_type_pairs = [
    ('10-6-79', 'notch'),
    ('7-28-92', 'notch'),
    ('8-10-74', 'curve'),
    ('9-13-90', 'notch'),
    ('10-6-92', 'notch'),
    ('10-9-80', 'notch'),
    ('11-2-97', 'fiducials_nisqually_1977'),
    ('1362', 'block')
]

# ## Preprocess each batch

# Process one example

# + jupyter={"outputs_hidden": true}
hsfm.batch.preprocess_images(
    '/Volumes/GoogleDrive/My Drive/hsfm-geomorph/data/fiducials/nagap/notch',
     image_metadata='output_data/10-6-79/targets.csv',
     image_directory='input_data/images_source/10-6-79',
     output_directory='output_data/10-6-79/preprocessed_images',
     qc=True
)
# -

# Process all subsets

date

for date, fiducial_type in date_fiducial_type_pairs:
    path_to_images = os.path.join(image_directory, date)
    path_to_fiducial_template = os.path.join(fiducial_dir,fiducial_type)

    targets_file_dir = os.path.join(output_data_dir, date)
    if not os.path.exists(targets_file_dir):
        os.makedirs(targets_file_dir)
    targets_file = os.path.join(targets_file_dir, 'targets.csv')
    image_df[image_df.Date == date.replace('-','/')].to_csv(targets_file)
    preprocess_output_directory = os.path.join(output_data_dir, date, 'preprocessed_images')
    print(f'Processing for {date}.')
    print(f'Source image directory: {path_to_images}')
    print(f'Fiducial marker template directory: {path_to_fiducial_template}')
    print(f'Targets csv path: {targets_file}')
    print(f'Image output directory: {preprocess_output_directory}')
    hsfm.batch.preprocess_images(
        path_to_fiducial_template,
        image_metadata=targets_file,
        image_directory=path_to_images,
        output_directory=preprocess_output_directory,
        qc=True
    )
    print()

# ## Create csv with heading info for each batch

hsfm.core.prepare_metashape_metadata(...)

#
