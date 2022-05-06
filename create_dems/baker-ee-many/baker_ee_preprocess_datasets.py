# Mt Baker
# EE Preprocess Datasets
# 
# This notebook walks through querying the EE database for aerial single frame images around Mt Baker.
# One image from each dataset, or "project" as the EE database calls them, is downloaded and fiducial
# proxy templates are generated manually through a locally hosted web app that is spawned the code here.
# Finally, all the images for each dataset are downloaded and preprocessed with the generated fiducial 
# proxy templates. The result of this notebook is a bunch of ready-for-SfM images.


# %%
import hipp.batch
import hipp.dataquery
import hsfm.batch
import matplotlib.pyplot as plt
import traceback
import geopandas as gpd
import contextily
from getpass import getpass
import os
# import holoviews as hv
# hv.extension('bokeh')

# %%
# Required inputs:
output_directory = '/data2/elilouis/timesift/baker-ee-many'
# strict bounds around Mt. Baker
xmin = -121.93271
xmax = -121.684585
ymin = 48.693888
ymax =  48.85011
# liberal bounds around Mt. Baker
# xmin = -122
# xmax = -121.5
# ymax =  49
# ymin = 48.5


# %%
# Login and get token from EarthExplorer API.
username = input()
apiKey = hipp.dataquery.EE_login(username, getpass())

startDate = '1901-01-01'
endDate   = '2022-01-01'

label     = 'test_download'

maxResults   = 50000

ee_results_df = hipp.dataquery.EE_pre_select_images(
    apiKey,
    xmin,ymin,xmax,ymax,
    startDate,endDate,
    maxResults   = maxResults
)

# %%
# ## Remove NAGAP images
ee_results_df = ee_results_df[~ee_results_df['project'].str.contains('NAG')]

# ## Filter out not available images
ee_results_df = ee_results_df[ee_results_df.hi_res_available=='Y']

# %%
ee_results_gdf = gpd.GeoDataFrame(ee_results_df, geometry = gpd.points_from_xy(
    ee_results_df['centerLon'], ee_results_df['centerLat'], crs='EPSG:4326'
))

# %%
# Look at photos
ax = ee_results_gdf.plot(column='project', legend=True, s=20)
leg = ax.get_legend()
leg.set_bbox_to_anchor((1, 1, 0, 0))
contextily.add_basemap(ax, crs=ee_results_gdf.crs)

# %%
# Download one image per project
for project, df in ee_results_df.groupby('project'):
    hipp.dataquery.EE_download_images_to_disk(
        apiKey, 
        list(df.entityId.head(2)), 
        output_directory = os.path.join(output_directory, f"single-image-downloads/{project}"),
        label='baker_bulk_download'
    )

# %%
# Group images based on entity ID
# Iterate through files, use filename to look up project in df table, save to new directory

import glob
path = os.path.join(output_directory, "raw_images")
fiducials_dir = os.path.join(output_directory, "fiducials")

for f in glob.glob(os.path.join(path, "*.tif")):
        entityId = f.split('/')[-1][:-4]
        data = ee_results_df[ee_results_df.entityId == entityId]
        os.renames(
            f,
            f.replace(
                f'raw_images/{entityId}.tif', 
                f'raw_images/{data.project.iloc[0]}/{entityId}.tif', 
            )
        )


# %%
# Create fiducials for each image dataset

for project in ee_results_df.project.unique():
    one_image_file = glob.glob(os.path.join(path, project, '*.tif'))[0]
    hipp.core.create_midside_fiducial_proxies_template(
        one_image_file,
        output_directory=os.path.join(fiducials_dir, project)
    )
    
    

# %%
# Run preprocessing batch flow for all EE datasets
# ToDo: I am duplicating the download of images here!! I should only be downloading one above to create fiducial templates
# Also, Everything about here should probably be in a separate script:
#   Script 1: Select projects, download 1 sample image from each, and create fiducials
#   Script 2: Preprocessing for all image sets 

for project in ee_results_df.project.unique():
    try:
        acquisition_date = ee_results_df[ee_results_df.project == project].acquisitionDate.iloc[0]
        [year, month, day] = acquisition_date.split('/')
        hsfm.batch.EE_pre_process_images(
            apiKey, 
            'all-ee-downloads', 
            [xmin, ymax, xmax, ymin], 
            project, 
            year, 
            month, 
            day,
            output_directory = output_directory,
            template_parent_dir = os.path.join(fiducials_dir, project)
        )
    except Exception as exc:
        print(traceback.format_exc())
        print(exc)
