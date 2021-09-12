# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: 'Python 3.8.5 64-bit (''hsfm'': conda)'
#     name: python3
# ---

# Trying to follow...
# https://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html

from glob import glob
import numpy as np
from scipy.io import loadmat
import rasterio as rio
import scipy.io
import rioxarray as rix
import geopandas as gpd
import matplotlib.pyplot as plt

# Create a clipped section of a large orthomosaic

# !gdal_translate -projwin 582332 5406614 583674 5405775 \
#     /data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/79_10.0/cluster0/1/orthomosaic_final.tif \
#     /data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/orthomosaic.tif

# Create a clipped section of a large DoD

# !gdal_translate -projwin 582332 5406614 583674 5405775 \
#     /data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/79_10.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.78_y+1.20_z+1.31_align_diff.tif \
#     /data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/dod.tif

# Create a clipped section of a large DEM

# !gdal_translate -projwin 582332 5406614 583674 5405775 \
#     /data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/79_10.0/cluster0/1/pc_align/spoint2point_bareground-trans_source-DEM_dem_align/spoint2point_bareground-trans_source-DEM_reference_dem_clipped_nuth_x+0.78_y+1.20_z+1.31_align.tif \
#     /data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/dem.tif

# Create a terrain ruggedness index tiff

# !gdaldem TRI \
#     -compute_edges \
#     /data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/dem.tif \
#     /data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/tri.tif

bands = [
    "/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/orthomosaic.tif",
    "/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/dod.tif",
    "/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/dem.tif",
    "/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/tri.tif"
]

# Open 4 layers

ortho = rix.open_rasterio(bands[0])
dod = rix.open_rasterio(bands[1], masked=True).rio.reproject_match(ortho)[0]
dem = rix.open_rasterio(bands[2], masked=True).rio.reproject_match(ortho)[0]
tri = rix.open_rasterio(bands[3], masked=True).rio.reproject_match(ortho)[0]

# Combine the alpha and greyscale bands in the orthomosaic by setting naNS

ortho_raster_values = ortho[0]
ortho_alpha_values = ortho[1]
ortho = ortho_raster_values.where(
    ortho_alpha_values == 255
)

type(ortho), type(dod), type(dem), type(tri)

ortho.values.shape, dod.values.shape, dem.values.shape, tri.values.shape

fix, axes = plt.subplots(2, 2, figsize=(20,12), sharex=True, sharey=True)
axes[0][0].imshow(ortho.values[::10, ::10], cmap='gray')
axes[0][1].imshow(dod.values[::10, ::10], cmap='PuOr')
axes[1][0].imshow(dem.values[::10, ::10], cmap='terrain')
axes[1][1].imshow(tri.values[::10, ::10], cmap='viridis')

from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

# Impute values for all 4 bands

ortho.values = imp.fit_transform(ortho.values)
ortho.values.shape

dod_fitted = imp.fit_transform(dod.values)
for i in range(0, dod.values.shape[1] - dod_fitted.shape[1]):
    dod_fitted = np.column_stack((dod_fitted, dod_fitted[:, -1]))
dod.values = dod_fitted
dod.values.shape

# Why are these failing? Imputer appears to be changing the shape of the data?

tri_fitted = imp.fit_transform(tri.values)
n_missing_cols = tri.values.shape[1] - tri_fitted.shape[1]
print(f'Adding {n_missing_cols} columns')
for i in range(0, n_missing_cols):
    tri_fitted = np.column_stack((tri_fitted, tri_fitted[:, -1]))
tri.values = tri_fitted
tri.values.shape

# + jupyter={"outputs_hidden": true}
dem_fitted = imp.fit_transform(dem.values)
n_missing_cols = dem.values.shape[1] - dem_fitted.shape[1]
print(f'Adding {n_missing_cols} columns')
for i in range(0, ):
    dem_fitted = np.column_stack((dem_fitted, dem_fitted[:, -1]))
dem.values = dem_fitted
dem.values.shape
# -

# Combine two layers into a single array

all_bands = np.dstack([ortho.values, dod.values, tri.values, dem.values])
all_bands.shape

# Load training data

from geocube.api.core import make_geocube

training_data_df = gpd.read_file("/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/training_data.geojson")

classes = {
    'water': 1,
    'forest': 2,
    'bareground': 3,
    'ice': 4,
}

training_data_df['key'] = training_data_df['id'].apply(classes.get)

training_data_df

# +
from geocube.api.core import make_geocube

result = make_geocube(
    training_data_df,
    measurements=["key"],
    resolution=(1, -1),
)
# -

result.key.rio.to_raster(
    "/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/training_data.tif"
)

# Reproject training data so our images are equal size and stuff

training_data = rix.open_rasterio(
    "/data2/elilouis/hsfm-geomorph/data/historical_land_cover_classification/training_data.tif"
).rio.reproject_match(ortho_raster_values)

plt.imshow(training_data.values[0])

training_data.plot()

# Classify

# replace nans in training data with 0

roi = training_data.values[0]
img = all_bands

roi.shape, img.shape

roi = np.nan_to_num(roi, 0)

labels = np.unique(roi[roi > 0])
print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                classes=labels))

X = img[roi > 0, :] 
y = roi[roi > 0]

X.shape, y.shape

# +
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500, oob_score=True)
rf = rf.fit(X, y)

# -

print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))

rf.feature_importances_

# Look at crosstabulation to see class confusion

# +
import pandas as pd

# Setup a dataframe -- just like R
df = pd.DataFrame()
df['truth'] = y
df['predict'] = rf.predict(X)

# Cross-tabulate predictions
print(pd.crosstab(df['truth'], df['predict'], margins=True))
# -

# Predict the rest of the image

img.shape

new_shape = (img.shape[0] * img.shape[1], img.shape[2])
img_as_array = img.reshape(new_shape)

print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

# +
# Now predict for each pixel
class_prediction = rf.predict(img_as_array)

# Reshape our classification map
class_prediction = class_prediction.reshape(img[:, :, 0].shape)
# -

# Visualize

class_prediction

# Visualize the predictions

prediction = ortho_raster_values.copy()
prediction.values = class_prediction

plt.imshow(ortho_raster_values.values, cmap='gray')

ortho_raster_values.plot(cmap='gray')

classes

flatui = ["#0000FF", "#008000", "#964B00", "#FFFFFF"]

prediction.plot(levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=flatui)

plt.imshow(prediction.values)
plt.colorbar()

plt.imshow(prediction.values, levels=[0.5, 1.5, 2.5, 3.5, 4.5], colors=flatui)


