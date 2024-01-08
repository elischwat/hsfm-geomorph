# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3.9.2 ('xdem')
#     language: python
#     name: python3
# ---

# Code included for option:
# * 1947, 1977/1979 mixed, 2015 based intervals

from datetime import datetime

import os
import geoutils as gu
import xdem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
from pprint import pprint
from rasterio.enums import Resampling

BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
print(f"retrieved base path: {BASE_PATH}")

# +
dem_fn_list1 = [
    os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/1947_09_14.tif"),
    os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/1977_09_27_clipped.tif"),
    os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/2015_09_01.tif")
]

dem_fn_list2 = [
    os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/1947_09_14.tif"),
    os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/1979_10_06_clipped.tif"),
    os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/2015_09_01.tif")
]

timestamps1 = ['1947_09_14', '1977_09_27', '2015_09_01']
timestamps2 = ['1947_09_14', '1979_10_06', '2015_09_01']

DATE_FILE_FORMAT = "%Y_%m_%d"
reference_dem_date = "2015_09_01"
reference_dem_date = datetime.strptime(
    reference_dem_date, 
    DATE_FILE_FORMAT
)

gcas_polygon_file = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/gcas.geojson")

output_file = "outputs/uncertainty_wholemountain.pcl"

RESAMPLING_RES = 5

FILTER_OUTLIERS = True
SIMPLE_FILTER = True
simple_filter_threshold = 50

VARIOGRAM_SUBSAMPLE = 100
VARIOGRAM_N_VARIOGRAMS = 10
PARALLELISM = 10
XSCALE_RANGE_SPLIT = [200]
MAX_LAG = 1000
# -

datetimes1 = [datetime.strptime(f, DATE_FILE_FORMAT) for f in timestamps1]
datetimes2 = [datetime.strptime(f, DATE_FILE_FORMAT) for f in timestamps2]

# # Create DEMCollection

# +
demcollection1 = xdem.DEMCollection.from_files(
    dem_fn_list1, 
    datetimes1, 
    reference_dem_date, 
    None, 
    RESAMPLING_RES,
    Resampling.cubic
)

demcollection2 = xdem.DEMCollection.from_files(
    dem_fn_list2, 
    datetimes2, 
    reference_dem_date, 
    None, 
    RESAMPLING_RES,
    Resampling.cubic
)

bounding_demcollection = xdem.DEMCollection.from_files(
    [dem_fn_list1[0], dem_fn_list1[-1]], 
    [datetimes1[0], datetimes1[-1]], 
    reference_dem_date, 
    None, 
    RESAMPLING_RES,
    Resampling.cubic
)
# -

print("demcollections generated")

_ = demcollection1.subtract_dems_intervalwise()
_ = demcollection2.subtract_dems_intervalwise()
_ = bounding_demcollection.subtract_dems_intervalwise()

print("DoDs generated")

# fig, axes = demcollection1.plot_ddems(figsize=(30, 10), vmin=-20, vmax=20, interpolation = "none")
# fig.savefig(os.path.join(plot_output_dir, "dod_gallery.png"))
# plt.show()

# fig, axes = demcollection2.plot_ddems(figsize=(30, 10), vmin=-20, vmax=20, interpolation = "none")
# fig.savefig(os.path.join(plot_output_dir, "dod_gallery.png"))
# plt.show()

# fig, axes = bounding_demcollection.plot_ddems(figsize=(30, 10), vmin=-20, vmax=20, interpolation = "none")
# fig.savefig(os.path.join(plot_output_dir, "dod_gallery.png"))
# plt.show()

gcas_vector = gu.Vector(gcas_polygon_file)

# +
# Collect the results as we create them
results_dict = {}

def clean_interval_string(interval):
    return interval.left.strftime("%y_%m_%d") + "__" + interval.right.strftime("%y_%m_%d")


# +
from uncertainty_helpers import uncertainty_analysis

print("Beginning to run uncertainty analysis...")

for ddem in demcollection1.ddems + demcollection2.ddems + bounding_demcollection.ddems:
    # try:
    results, figs = uncertainty_analysis(
        ddem,
        gcas_vector,
        subsample = VARIOGRAM_SUBSAMPLE,
        n_variograms = VARIOGRAM_N_VARIOGRAMS,
        xscale_range_split = XSCALE_RANGE_SPLIT,
        parallelism=PARALLELISM,
        maxlag=MAX_LAG,
        FILTER_OUTLIERS = FILTER_OUTLIERS,
        SIMPLE_FILTER = SIMPLE_FILTER,
        simple_filter_threshold = simple_filter_threshold
    )
    interval_string = clean_interval_string(ddem.interval)
    # figs[0].savefig(os.path.join(plot_output_dir, f"dod_uncertainty_static_areas_{interval_string}.png"))
    # figs[1].savefig(os.path.join(plot_output_dir, f"dod_uncertainty_empirical_variogram_{interval_string}.png"))
    # figs[2].savefig(os.path.join(plot_output_dir, f"dod_uncertainty_fit_variogram_{interval_string}.png"))
    pprint(results, width=1)
    results_dict[results["Interval"]] = results
    # except Exception as exc:
        # print(f"Failed on ddem: {ddem.interval}")
        # print(exc)

print("Uncertainty analyses completed...")

# -

from scipy import stats

# +
results_df = pd.DataFrame(results_dict).transpose().reset_index(drop=True)
results_df['Start Date'] = results_df['Interval'].apply(lambda x: x.left)
results_df['End Date'] = results_df['Interval'].apply(lambda x: x.right)
results_df['NMAD'] = pd.to_numeric(results_df['NMAD'])
results_df['Mean'] = pd.to_numeric(results_df['Mean'])
results_df['RMSE'] = pd.to_numeric(results_df['RMSE'])
results_df['Range'] = pd.to_numeric(results_df['Range'])
results_df['Sill'] = pd.to_numeric(results_df['Sill'])

results_df['StdDev'] = pd.to_numeric(results_df['StdDev'])
results_df['90% CI'] = results_df.apply(lambda row: stats.norm.interval(0.90, loc=row['Mean'], scale=row['StdDev']), axis=1)
results_df['90% CI Lower Bound'] = pd.to_numeric(results_df['90% CI'].apply(lambda x: x[0]))
results_df['90% CI Upper Bound'] = pd.to_numeric(results_df['90% CI'].apply(lambda x: x[1]))

results_df
# -

results_df['bounding'] = results_df['Interval'].apply(lambda x: x.left == datetimes1[0] and x.right == datetimes1[-1])
results_df

# +
chart = alt.Chart(results_df.drop(columns="Interval")).transform_filter(
   alt.datum.bounding == False
).mark_bar(
    strokeWidth = 3,
    stroke="white",
).encode(
    alt.X("Start Date:T"),
    alt.X2("End Date:T"),
    alt.Y(alt.repeat("row"), type='quantitative'),
).properties(
    # width=200,
    height=150
).repeat(
    row=['NMAD', 'Mean', 'RMSE', 'Range', 'Sill', 'StdDev', '90% CI Lower Bound', '90% CI Upper Bound']
)

chart_bounding = alt.Chart(results_df.drop(columns="Interval")).transform_filter(
   alt.datum.bounding == True
).mark_bar(
    strokeWidth = 3,
    stroke="white",
).encode(
    alt.X("Start Date:T"),
    alt.X2("End Date:T"),
    alt.Y(alt.repeat("row"), type='quantitative'),
).properties(
    # width=200,
    height=150
).repeat(
    row=['NMAD', 'Mean', 'RMSE', 'Range', 'Sill', 'StdDev', '90% CI Lower Bound', '90% CI Upper Bound']
)

chart
# -

chart_bounding

results_df
print("Saving uncertainty results...")
results_df.to_pickle(output_file)


