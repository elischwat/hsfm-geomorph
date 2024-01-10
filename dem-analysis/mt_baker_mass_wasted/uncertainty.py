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

from datetime import datetime
from pathlib import Path
import glob
import os
import geoutils as gu
import xdem
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import pandas as pd
import altair as alt
from pprint import pprint
from rasterio.enums import Resampling
import copy
import json 

# # Inputs

# * Inputs are written in a JSON.
# * The inputs file is specified by the `HSFM_GEOMORPH_INPUT_FILE` env var
# * One input may be overriden with an additional env var - `RUN_LARGER_AREA`. If this env var is set to "yes" or "no" (exactly that string, it will be used. If the env var is not set, the params file is used to fill in this variable. If some other string is set, a failure is thrown).

# If you use the arg, you must run from CLI like this
#
# ```
# HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_mazama.html
# ```

BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
print(f"retrieved base path: {BASE_PATH}")

if os.environ.get('HSFM_GEOMORPH_INPUT_FILE'):
    json_file_path = os.environ['HSFM_GEOMORPH_INPUT_FILE']
else:
    json_file_path = 'inputs/rainbow_inputs.json'
print(f"retrieved input file: {json_file_path}")

with open(json_file_path, 'r') as j:
     params = json.loads(j.read())

# +
VALLEY_BOUNDS_NAME = params["inputs"]["valley_name"]
dems_path = os.path.join(BASE_PATH, params["inputs"]["dems_path"])
gcas_polygon_file = os.path.join(BASE_PATH, params["uncertainty"]["gcas_polygon_file"])
valley_bounds_file = os.path.join(BASE_PATH, params["inputs"]["valley_bounds_file"])
plot_output_dir = os.path.join(BASE_PATH, params["inputs"]["plot_output_dir"])
output_file = params["inputs"]["uncertainty_file"]
output_file_largerarea = params["inputs"]["uncertainty_file_largearea"]
TO_DROP = params["inputs"]["TO_DROP"]
TO_DROP_LARGERAREA = params["inputs"]["TO_DROP_LARGER_AREA"]
TO_COREGISTER = params["inputs"]["TO_COREGISTER"]
DATE_FILE_FORMAT = params['inputs']['strip_time_format']
FILTER_OUTLIERS = params['inputs']['FILTER_OUTLIERS']
SIMPLE_FILTER = params['inputs']['SIMPLE_FILTER']
simple_filter_threshold = params['inputs']['simple_filter_threshold']

reference_dem_date = datetime.strptime(
    params['inputs']['reference_dem_date'], 
    DATE_FILE_FORMAT
)

if os.environ.get('RUN_LARGER_AREA'):
    print("RUN_LARGER_AREA env var read.")
    if os.environ['RUN_LARGER_AREA'] == "yes":
        print("Running larger area")
        RUN_LARGER_AREA = True
    elif os.environ['RUN_LARGER_AREA'] == "no":
        print("NOT running larger area")
        RUN_LARGER_AREA = False
    else:
        raise ValueError("Env Var RUN_LARGER_AREA set to an incorrect value. Cannot proceed.")
else:
    RUN_LARGER_AREA = params['inputs']['RUN_LARGER_AREA']

VARIOGRAM_SUBSAMPLE = params["uncertainty"]["VARIOGRAM_SUBSAMPLE"]
VARIOGRAM_N_VARIOGRAMS = params["uncertainty"]["VARIOGRAM_N_VARIOGRAMS"]
PARALLELISM = params["uncertainty"]["PARALLELISM"]
XSCALE_RANGE_SPLIT = params["uncertainty"]["XSCALE_RANGE_SPLIT"]
MAX_LAG = params["uncertainty"]["MAX_LAG"]
RESAMPLING_RES = params["uncertainty"]["RESAMPLING_RES"]
# -

VARIOGRAM_SUBSAMPLE, \
VARIOGRAM_N_VARIOGRAMS, \
PARALLELISM, \
XSCALE_RANGE_SPLIT, \
MAX_LAG, \
RESAMPLING_RES

if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir, exist_ok=True)

# # Get DEM file paths

# +
dem_fn_list = glob.glob(os.path.join(dems_path, "*.tif"))
dem_fn_list = sorted(dem_fn_list)

if RUN_LARGER_AREA:
    dem_fn_list = [f for f in dem_fn_list if Path(f).stem not in TO_DROP_LARGERAREA]
else:
    dem_fn_list = [f for f in dem_fn_list if Path(f).stem not in TO_DROP]
dem_fn_list
# -

dem_fn_list = [f for f in dem_fn_list if 'unaligned' not in f]
dem_fn_list

datetimes = [datetime.strptime(Path(f).stem, DATE_FILE_FORMAT) for f in dem_fn_list]
datetimes

# # Open Valley Bounds Geometry

valley_bounds = gu.Vector(valley_bounds_file)
uncertainty_valley_bounds_vect = valley_bounds.query(f"name == '{VALLEY_BOUNDS_NAME}' and purpose=='uncertainty'")
uncertainty_valley_bounds_vect.ds

# # Create DEMCollection

demcollection_uncertainty = xdem.DEMCollection.from_files(
    dem_fn_list, 
    datetimes, 
    reference_dem_date, 
    uncertainty_valley_bounds_vect, 
    RESAMPLING_RES,
    Resampling.cubic
)

if TO_COREGISTER:
    for i in range(0, len(demcollection_uncertainty.dems)-1):
        early_dem = demcollection_uncertainty.dems[i]
        late_dem = demcollection_uncertainty.dems[i+1]

        nuth_kaab = xdem.coreg.NuthKaab()
        # Order with the future as reference
        nuth_kaab.fit(late_dem.data, early_dem.data, transform=late_dem.transform)

        # Apply the transformation to the data (or any other data)
        aligned_ex = nuth_kaab.apply(early_dem.data, transform=early_dem.transform)

        print(F"For DEM {early_dem.datetime}, transform is {nuth_kaab.to_matrix()}")

        early_dem.data = np.expand_dims(aligned_ex, axis=0)

_ = demcollection_uncertainty.subtract_dems_intervalwise()

# # Create Bounding DEMCollection

# +
bounding_demcollection_uncertainty = xdem.DEMCollection(
    [demcollection_uncertainty.dems[0], demcollection_uncertainty.dems[-1]],
    [demcollection_uncertainty.timestamps[0], demcollection_uncertainty.timestamps[-1]],
)

_ = bounding_demcollection_uncertainty.subtract_dems_intervalwise()
# -

print("demcollections generated")

# # Calculate Uncertainty

# ## Open ground control polygons

gcas_vector = gu.Vector(gcas_polygon_file)

# ## Define function to perform an uncertainty analysis:
# * Plot ground control area DH
# * Sample dataset and plot empirical variogram
# * Fit spherical model and plot empirical variogram + fitted model
# * Print comprehensive statistics

from uncertainty_helpers import uncertainty_analysis

# Collect the results as we create them
results_dict = {}


def clean_interval_string(interval):
    return interval.left.strftime("%y_%m_%d") + "__" + interval.right.strftime("%y_%m_%d")


# +
def run_analysis_plot_and_return_results(ddem):
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
    figs[0].savefig(os.path.join(plot_output_dir, f"dod_uncertainty_static_areas_{interval_string}.png"))
    figs[1].savefig(os.path.join(plot_output_dir, f"dod_uncertainty_empirical_variogram_{interval_string}.png"))
    figs[2].savefig(os.path.join(plot_output_dir, f"dod_uncertainty_fit_variogram_{interval_string}.png"))
    pprint(results, width=1)
    return results

print("Beginning to run analysis part 1...")    
for i,ddem in enumerate(demcollection_uncertainty.ddems):
    results = run_analysis_plot_and_return_results(ddem)
    results['bounding'] = False
    results_dict[results["Interval"]] = results
    print(f"{i} of {len(demcollection_uncertainty.ddems)} analyses completed")    

print("Beginning to run analysis part 2...")    
for i,ddem in enumerate(bounding_demcollection_uncertainty.ddems):
    results = run_analysis_plot_and_return_results(ddem)
    results['bounding'] = True
    results_dict[results["Interval"]] = results
    print(f"{i} of {len(bounding_demcollection_uncertainty.ddems)} analyses completed")    
# -

print("Beginning to analyze results...")    
# ### Analyze all uncertainty results

from scipy import stats

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

alt.Chart(results_df.query("bounding == False").drop(columns=["Interval", "90% CI"])).mark_bar(
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
# chart.save(os.path.join(plot_output_dir, "uncertainty_results.png"), scale_factor=2.0)

alt.Chart(results_df.query("bounding == True").drop(columns=["Interval", "90% CI"])).mark_bar(
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
# chart.save(os.path.join(plot_output_dir, "uncertainty_results.png"), scale_factor=2.0)

print("Saving results...")    
if RUN_LARGER_AREA:
    Path(os.path.dirname(output_file_largerarea)).mkdir(parents=True, exist_ok=True)
    results_df.to_pickle(output_file_largerarea)
else:
    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
    results_df.to_pickle(output_file)


