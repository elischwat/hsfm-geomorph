# Mt Baker
# NAGAP Preprocess datasets
# 
# This notebook querys for NAGAP images, downloads them, and preprocesses them. It is assumed that you already have
# the necessary fiducial proxy template files. These are available in HIPP repo (github.com/friedrichknuth/hipp).
# The result of this notebook is a bunch of ready-for-SfM images.

from hsfm.pipeline import NAGAPPreprocessingPipeline

# Required inputs:
output_directory = '/data2/elilouis/timesift/baker-ee-many'
preprocessed_ee_image_directory = '/data2/elilouis/timesift/baker-ee-many/baker-ee-many'
fiducial_templates_directory = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/fiducials/nagap/"
bounds_for_nagap_image_search = (-121.94, 48.84, -121.70, 48.70)
nagap_image_info_fn = '/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata_updated_manual.csv'

# Run NAGAP preprocessing 
preprocess_pipeline = NAGAPPreprocessingPipeline(
    output_directory,
    fiducial_templates_directory = fiducial_templates_directory,
    nagap_images_csv_path = nagap_image_info_fn,
    bounds = bounds_for_nagap_image_search
)
preprocess_pipeline.run()