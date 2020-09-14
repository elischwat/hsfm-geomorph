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

import bare
import hsfm

targets_file = 'targets_kautz_watershed_1977.csv'
thumbnail_directory = 'input_data/thumbnails'
fiducial_template_dir = '/home/elilouis/hsfm/examples/input_data/fiducials/nagap/notch'
output_directory = 'input_data/images'


# # Pre-process imagery

# ## Download thumbnails

# cat $targets_file | head -2

hsfm.batch.download_images_to_disk(targets_file, 
                                   output_directory=thumbnail_directory,
                                   image_type='pid_tn') # pid_tiff, pid_tn, pid_jpeg

# ## Examine a thumbnail

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
imgplot = plt.imshow(mpimg.imread('input_data/thumbnails/NAGAP_77V6_263.tif'))
plt.gcf().set_size_inches(20,20)
plt.xlim
plt.show()

# Looks like the fiducal markers don't match an existing pattern...

# ## Detect fiducial markers, crop, and enhance contrast

hsfm.batch.preprocess_images(fiducial_template_dir,
                             camera_positions_file_name=targets_file,
                             output_directory=output_directory,
                             qc=True)


