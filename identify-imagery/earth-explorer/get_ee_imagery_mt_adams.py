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

# %load_ext autoreload
# %autoreload 2
import hipp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import glob
import hipp
import os
import numpy as np
import hsfm


project = "18900"

apiKey = hipp.dataquery.EE_login(input(), input())

# Bounds around Mt. Baker
xmin = -121.94
xmax = -121.70
ymin = 48.7
ymax = 48.84

startDate = '1901-01-01'
endDate   = '2022-01-01'

label     = project
output_directory = '/data2/elilouis/earth_explorer_dem_mt_baker/'

maxResults   = 50000

scenes_df = hipp.dataquery.EE_sceneSearch(
    apiKey,
    xmin,ymin,xmax,ymax,
    startDate, endDate,
    maxResults = maxResults
)
df = hipp.dataquery.EE_filterSceneRecords(scenes_df)

df[df['project']==project]

src = df[df['project']==project]

entityIds = list(src['entityId'])

import hipp

images_directory, calibration_reports_directory = hipp.dataquery.EE_downloadImages(
    apiKey,
    entityIds
)


