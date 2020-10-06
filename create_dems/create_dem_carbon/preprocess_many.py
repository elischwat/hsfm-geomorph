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

data_dir = '/home/elilouis/hsfm-geomorph/data'
license_path = '/home/elilouis/hsfm/uw_agisoft.lic'

hsfm.metashape.authentication(license_path)

# +
image_directory = 'input_data/images_source'

targets_file = '../../identify-imagery/targets_carbon_all_dates.csv'

image_df = pd.read_csv(targets_file)
# -

# ## Download all tiffs

# + jupyter={"outputs_hidden": true}
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


# -

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

# ## Create fiducial marker templates
# ## ~Assign directories to fiducial markers for each date~

# <span style="color:red">MAY REQUIRE MANUAL INTERVENTION</span>

for date_dir in os.listdir(image_directory):
    fiducial_template_dir = os.path.join('fiducials/', date_dir)
    img_name = os.listdir(os.path.join(image_directory, date_dir))[0]
    img_path = os.path.join(os.path.join(image_directory, date_dir), img_name)
    print(f'Creating fiducials with image {img_path} for {date_dir} to output_directory {fiducial_template_dir}.')

    img = mpimg.imread(img_path)
    hsfm.utils.create_fiducials(img, output_directory=fiducial_template_dir)


# ## Preprocess each batch

hsfm.batch.preprocess_images(...)

# ## Create csv with heading info for each batch

hsfm.core.prepare_metashape_metadata(...)

#
