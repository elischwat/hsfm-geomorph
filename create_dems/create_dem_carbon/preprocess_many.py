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
license_path = 'create_dems/create_dem_carbon/uw_agisoft.lic'

# + [markdown] jupyter={"outputs_hidden": true}
# hsfm.metashape.authentication(license_path)

# +
image_directory = 'input_data/images_source'
output_data_dir = 'output_data'

targets_file = '../../identify-imagery/targets_carbon_all_dates.csv'

image_df = pd.read_csv(targets_file)


# -

# ## Download all tiffs

# batch_number = 1
# for date, data in image_df.groupby('Date'):
#     data.to_csv('temp.csv', index=None)
#     print(f'Downloading batch {batch_number}')
#     hsfm.batch.download_images_to_disk(
#         'temp.csv', 
#         output_directory=os.path.join(image_directory,  str(date).replace('/','-'))
#     )
#     # # # # # !rm temp.csv
#     batch_number = batch_number + 1

# ## What images were meant to be downloaded that were not?

def list_all_tif_files_in_dir(directory):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    ls = []
    for x in os.walk(directory):
        ls.append(flatten(list(x)[1:]))
    return [token for token in flatten(ls) if token.endswith('.tif')]


downloaded_images = list_all_tif_files_in_dir(image_directory)

len(image_df.fileName), len(downloaded_images), set(downloaded_images).difference(set(image_df.fileName+'.tif')), set(image_df.fileName+'.tif').difference(set(downloaded_images))

# ## Create fiducial marker templates for each date

# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

fiducial_dir = os.path.join(data_dir, 'fiducials/nagap')

fiducial_dir

# + [markdown] jupyter={"outputs_hidden": true}
# for date_dir in os.listdir(image_directory):
#     fiducial_template_dir = os.path.join('fiducials/', date_dir)
#     img_name = os.listdir(os.path.join(image_directory, date_dir))[0]
#     img_path = os.path.join(os.path.join(image_directory, date_dir), img_name)
#     print(f'Creating fiducials with image {img_path} for {date_dir} to output_directory {fiducial_template_dir}.')
#
#     img = mpimg.imread(img_path)
#     hsfm.utils.create_fiducials(img, output_directory=fiducial_template_dir)
# -


# ## Or just assign fiducial template directories directly for each date

date_fiducial_type_pairs = [
    ('10-6-79', 'notch'),
    ('7-28-92', 'notch'),
    ('8-10-74', 'curve'),
    ('9-13-90', 'fiducials_carbon_9-13-90'),
    ('10-6-92', 'notch'),
    ('10-9-80', 'notch'),
    ('11-2-97', 'fiducials_nisqually_1977'), #<-- wrong date! These are from 1977 according to roll/filename
    ('1362', 'block')
]


# ## Preprocess each batch

def preprocess_images_for_date(date, fiducial_type, **kwargs):
    """
    Process images associated with one subset date. 
    Appropriate kwargs:
    invisible_fiducial='right',
    """
    path_to_images = os.path.join(image_directory, date)
    path_to_fiducial_template = os.path.join(fiducial_dir,fiducial_type)
    targets_file_dir = os.path.join(output_data_dir, date)
    if not os.path.exists(targets_file_dir):
        os.makedirs(targets_file_dir)
    targets_file = os.path.join(targets_file_dir, 'targets.csv')
    image_df[image_df.Date == date.replace('-','/')].to_csv(targets_file)
    preprocess_output_directory = os.path.join(targets_file_dir, 'preprocessed_images')

    print(f'Processing images for {date}.')
    print(f'Using source image directory: {path_to_images}')
    print(f'Using fiducial marker template directory: {path_to_fiducial_template}')
    print(f'Outputting image subset csv list to: {targets_file}')
    print(f'Outputting preprocessed images to: {preprocess_output_directory}')
    hsfm.batch.preprocess_images(
        path_to_fiducial_template,
        image_metadata=targets_file,
        image_directory=path_to_images,
        output_directory=preprocess_output_directory,
        qc=True,
        **kwargs
    )
    print()


# Process problematic individual sets

# #### Process 10-6-92
#
# 1. invisible_fiducial = 'right'
# 2. Fails because cropping distance is too small - anything larger than 10758 fails.

preprocess_images_for_date(
    '10-6-92',
    'notch',
    invisible_fiducial='right',
    crop_from_pp_dist = 10758
)

# #### Process 11-2-97
#
# Problems:
#     
# 1. Incorrect date - real date is 11-2-77.
# 2. Missing top fiducials, so I specify invisible_fiducial='top'
# 3. Larger angle threshold allows less manual intervention

preprocess_images_for_date(
    '11-2-97',
    'fiducials_nisqually_1977',
    invisible_fiducial='top',
    angle_threshold=0.7
)

# #### Process 1362
#
# Problems:
#
# 1. Files are not labelled with correct date.
# 2. 

preprocess_images_for_date('1362', 'block')

# #### Process 8-10-74

preprocess_images_for_date('8-10-74', 'curve', angle_threshold=0.3)

# #### Process all subsets

for date, fiducial_type in date_fiducial_type_pairs:
    preprocess_images_for_date(date, fiducial_type)

# ## Create csv with heading info for each batch

# Problems encountered:
# 1. For date 8-10-74, targets.csv does not have lat/long columns - can I retrieve from KML files?

dates = [
    '10-6-79',
    '7-28-92',
    '8-10-74',
    '9-13-90',
    '10-6-92',
    '10-9-80',
    '11-2-97',
    '1362'
]

srtm_reference_dem_warped = f'{data_dir}/reference_dem/SRTM3/SRTM3_warped.tif'
for date in dates:
    targets_file = os.path.join(output_data_dir, date, 'targets.csv')
    appropriate_output_directory = os.path.join(output_data_dir, date)
    print(f'Using targets file:\t\t{targets_file}')
    print(f'Placing heading info to:\t{appropriate_output_directory}')
    print()
    hsfm.core.prepare_metashape_metadata(
        targets_file, 
        reference_dem=srtm_reference_dem_warped,
        output_directory=appropriate_output_directory
    )

# ## How many source images do not have preprocessed images?

# ls $image_directory

# ls $output_data_dir

import glob

for date in dates:
    input_image_list = glob.glob(os.path.join(image_directory, date, "*.tif"))
    output_image_list = glob.glob(os.path.join(output_data_dir, date, "preprocessed_images", "*.tif"))
    print(f'Input image count: {len(input_image_list)}')
    print(f'Output image count: {len(output_image_list)}')
    print(f'Output less Input count: {len(output_image_list) - len(input_image_list)}')
    print()
