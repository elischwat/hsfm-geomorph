# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: hsfm
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
import geoutils as gu
import xdem
from pprint import pprint
import altair as alt
from rasterio.enums import Resampling
import json 
import seaborn as sns

# %% [markdown]
# ## Inputs

# %% [markdown]
# * Inputs are written in a JSON.
# * The inputs file is specified by the `HSFM_GEOMORPH_INPUT_FILE` env var
# * One input may be overriden with an additional env var - `RUN_LARGER_AREA`. If this env var is set to "yes" or "no" (exactly that string, it will be used. If the env var is not set, the params file is used to fill in this variable. If some other string is set, a failure is thrown).

# %% [markdown]
# If you use the arg, you must run from CLI like this
#
# ```
# HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_mazama.html
# ``

BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
print(f"retrieved base path: {BASE_PATH}")

# %%
# Or set an env arg:
if os.environ.get('HSFM_GEOMORPH_INPUT_FILE'):
    json_file_path = os.environ['HSFM_GEOMORPH_INPUT_FILE']
else:
    json_file_path = 'inputs/rainbow_inputs.json'

# %%
with open(json_file_path, 'r') as j:
     params = json.loads(j.read())

# %%
params

# %%
# Read inputs from params
valley_name = params['inputs']['valley_name']
TO_DROP = params['inputs']['TO_DROP']
TO_DROP_LARGER_AREA = params['inputs']['TO_DROP_LARGER_AREA']
TO_COREGISTER = params['inputs']['TO_COREGISTER']
SAVE_DDEMS = params['inputs']['SAVE_DDEMS']
EROSION_BY_DATE = params['inputs']['EROSION_BY_DATE']
INTERPOLATE = params['inputs']['INTERPOLATE']
FILTER_OUTLIERS = params['inputs']['FILTER_OUTLIERS']
glacier_polygons_file = os.path.join(BASE_PATH, params['inputs']['glacier_polygons_file'])
dems_path = os.path.join(BASE_PATH, params["inputs"]["dems_path"])
valley_bounds_file = os.path.join(BASE_PATH, params["inputs"]["valley_bounds_file"])
strip_time_format = params['inputs']['strip_time_format']
plot_output_dir = os.path.join(BASE_PATH, params["inputs"]["plot_output_dir"])
uncertainty_file = params['inputs']['uncertainty_file']
uncertainty_file_largerarea = params["inputs"]["uncertainty_file_largearea"]
SIMPLE_FILTER = params['inputs']['SIMPLE_FILTER']
simple_filter_threshold = params['inputs']['simple_filter_threshold']

plot_figsize = params['inputs']['plot_figsize']
plot_vmin = params['inputs']['plot_vmin']
plot_vmax = params['inputs']['plot_vmax']
MASK_GLACIER_SIGNALS = params['inputs']['MASK_GLACIER_SIGNALS']
MASK_EXTRA_SIGNALS = params['inputs']['MASK_EXTRA_SIGNALS']


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


dem_target_resolution = params["inputs"]['dem_target_resolution']

interpolation_max_search_distance = params['inputs']['interpolation_max_search_distance']

if EROSION_BY_DATE:
    erosion_polygon_file = os.path.join(BASE_PATH, params['inputs']['erosion_by_date_polygon_file'])
else:
    erosion_polygon_file = os.path.join(BASE_PATH, params['inputs']['erosion_polygon_file'])

# Read output inputs from params
erosion_polygons_cropped_by_glaciers_output_file = params['outputs']['erosion_polygons_cropped_by_glaciers_output_file']
dods_output_path = params['outputs']['dods_output_path']

reference_dem_date = datetime.strptime(
    params['inputs']['reference_dem_date'], 
    strip_time_format
)

# %%
if RUN_LARGER_AREA:
    uncertainty_df = pd.read_pickle(uncertainty_file_largerarea)
else:
    uncertainty_df = pd.read_pickle(uncertainty_file)
uncertainty_df.head()

# %%
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir, exist_ok=True)

# %% [markdown]
# ## Get DEM file paths

# %%
dem_fn_list = glob.glob(os.path.join(dems_path, "*.tif"))
dem_fn_list = sorted(dem_fn_list)

if RUN_LARGER_AREA:
    dem_fn_list = [f for f in dem_fn_list if Path(f).stem not in TO_DROP_LARGER_AREA]
else:
    dem_fn_list = [f for f in dem_fn_list if Path(f).stem not in TO_DROP]
dem_fn_list

# %%
dem_fn_list = [f for f in dem_fn_list if 'unaligned' not in f]
dem_fn_list

# %%
datetimes = [datetime.strptime(Path(f).stem, strip_time_format) for f in dem_fn_list]
datetimes

# %% [markdown]
# ## Open valley bounds polygons

# %%
valley_bounds = gu.Vector(valley_bounds_file)
if RUN_LARGER_AREA:
    valley_bounds_vect = valley_bounds.query(f"name == '{valley_name}' and purpose=='analysis large'")
else:
    valley_bounds_vect = valley_bounds.query(f"name == '{valley_name}' and purpose=='analysis'")

# %% [markdown]
# ## Create DEMCollection

# %%
demcollection = xdem.DEMCollection.from_files(
    dem_fn_list, 
    datetimes, 
    reference_dem_date, 
    valley_bounds_vect, 
    dem_target_resolution,
    resampling = Resampling.cubic
)

# %% [markdown]
# ## Open glacier polygons

# %%
glaciers_gdf = gpd.read_file(glacier_polygons_file).to_crs(demcollection.reference_dem.crs)
glaciers_gdf['date'] = glaciers_gdf['year'].apply(lambda x: datetime.strptime(x, strip_time_format))

# %% [markdown]
# ## Plot DEMs

# %%
fig, axes = demcollection.plot_dems(hillshade=True, interpolation = "none", figsize=plot_figsize)
fig.savefig(os.path.join(plot_output_dir, "dem_gallery.png"))
plt.suptitle("DEM Gallery")
plt.show()

# %% [markdown]
# ## Coregister DEMs or Do Not

# %%

if TO_COREGISTER:
    for i in range(0, len(demcollection.dems)-1):
        early_dem = demcollection.dems[i]
        late_dem = demcollection.dems[i+1]

        nuth_kaab = xdem.coreg.NuthKaab()
        # Order with the future as reference
        nuth_kaab.fit(late_dem.data, early_dem.data, transform=late_dem.transform, 
            inlier_mask = ~gu.Vector(glaciers_gdf).create_mask(early_dem).squeeze()
        )

        # Apply the transformation to the data (or any other data)
        aligned_ex = nuth_kaab.apply(early_dem.data, transform=early_dem.transform)

        print(F"For DEM {early_dem.datetime}, transform is {nuth_kaab.to_matrix()}")

        early_dem.data = np.expand_dims(aligned_ex, axis=0)

# %% [markdown]
# ## Subtract DEMs/Create DoDs

# %%
_ = demcollection.subtract_dems_intervalwise()
# _ = demcollection_large.subtract_dems_intervalwise()

# %% [markdown]
# ## Plot DoDs (pre processing)

# %%
fig, axes = demcollection.plot_ddems(
    figsize=plot_figsize, vmin=-30, vmax=30, 
    interpolation = "none", 
    plot_outlines=False,
    hillshade=True,
    cmap_alpha=0.15
)
plt.suptitle("dDEM Gallery")
fig.savefig(os.path.join(plot_output_dir, "dod_gallery_preprocessing.png"))
plt.show()

# %%
fig, axes = demcollection.plot_ddems(
    figsize=plot_figsize, vmin=plot_vmin, vmax=plot_vmax, 
    interpolation = "none", 
    plot_outlines=False,
    hillshade=True,
    cmap_alpha=0.15
)
plt.suptitle("dDEM Gallery")
fig.savefig(os.path.join(plot_output_dir, "dod_gallery_preprocessing.png"))
plt.show()

# %% [markdown]
# ## Save DoDs without Cropping stuff

# %%
if SAVE_DDEMS:
    # Save all interval dDEMs
    os.makedirs(dods_output_path, exist_ok=True)

    for ddem in demcollection.ddems:
        startt = ddem.start_time.strftime(strip_time_format)
        endt = ddem.end_time.strftime(strip_time_format)
        if RUN_LARGER_AREA:
            fn = f"{startt}_to_{endt}_largerarea_asis.tif"
        else:
            fn = f"{startt}_to_{endt}_asis.tif"
        fn = os.path.join(dods_output_path, fn)
        print(fn)
        ddem_copy = ddem.copy()
        filled_data = ddem_copy.interpolate(
            method="linear", 
            reference_elevation=demcollection.reference_dem, 
            max_search_distance=interpolation_max_search_distance
        )
        ddem_copy.set_filled_data()
        ddem_xr = ddem_copy.to_xarray()
        ddem_xr.data = ddem_copy.data.filled(np.nan)
        ddem_xr.rio.to_raster(fn)

    # Save bounding dDEM

    bounding_ddem = xdem.dDEM(  
        demcollection.dems[-1] - demcollection.dems[0],
        demcollection.timestamps[0], 
        demcollection.timestamps[-1]
    )
    filled_data = bounding_ddem.interpolate(
        method="linear", 
        reference_elevation=demcollection.reference_dem, 
        max_search_distance=interpolation_max_search_distance
    )
    bounding_ddem.set_filled_data()
    bounding_ddem_xr = bounding_ddem.to_xarray()
    bounding_ddem_xr.data = bounding_ddem.data.filled(np.nan)
    startt = pd.Timestamp(bounding_ddem.start_time).strftime(strip_time_format)
    endt = pd.Timestamp(bounding_ddem.end_time).strftime(strip_time_format)
    if RUN_LARGER_AREA:
        fn = f"{startt}_to_{endt}_largerarea_asis.tif"
    else:
        fn = f"{startt}_to_{endt}_asis.tif"
    fn = os.path.join(dods_output_path, fn)
    print(fn)
    # bounding_ddem_copy = bounding_ddem.copy()
    bounding_ddem_xr.rio.to_raster(fn)

# %% [markdown]
# ## Mask Glacier Signals

# %%
if MASK_GLACIER_SIGNALS:
    for ddem in demcollection.ddems:
        ddem
        relevant_glaciers_gdf = glaciers_gdf[glaciers_gdf['date'].isin([ddem.interval.left, ddem.interval.right])]
        relevant_glaciers_mask = gu.Vector(relevant_glaciers_gdf).create_mask(ddem).squeeze()
        ddem.data.mask = np.logical_or(ddem.data.mask, relevant_glaciers_mask)

# %% [markdown]
# ## Filter outliers

# %%
if FILTER_OUTLIERS:
    if SIMPLE_FILTER:
        for dh in demcollection.ddems:
            dh.data = np.ma.masked_where(np.abs(dh.data) > simple_filter_threshold, dh.data)
    else:
        for dh in demcollection.ddems:
            all_values_masked = dh.data.copy()
            all_values = all_values_masked.filled(np.nan)
            low = np.nanmedian(all_values) - 4*xdem.spatialstats.nmad(all_values)
            high = np.nanmedian(all_values) + 4*xdem.spatialstats.nmad(all_values)
            print(np.nanmax(dh.data))
            print(np.nanmin(dh.data))
            print(dh.interval)
            print(low)
            print(high)
            all_values_masked = np.ma.masked_greater(all_values_masked, high)
            all_values_masked = np.ma.masked_less(all_values_masked, low)
            dh.data = all_values_masked
            print(np.nanmax(dh.data))
            print(np.nanmin(dh.data))
            print()

# %% [markdown]
# ## Prepare erosion polygons

# %% [markdown]
# ### Load erosion polygons

# %%
erosion_vector = gu.Vector(erosion_polygon_file)
erosion_vector.ds = erosion_vector.ds.to_crs(demcollection.reference_dem.crs)
erosion_vector.ds.head(3)

# %% [markdown]
# ### Subtract glacier polygons from erosion polygons
#
# Only applies if not EROSION_BY_DATE
#
# For each dDEM time interval, get the two relevant glacier polygons, and subtract them from each erosion polygon, so that each erosion polygon multiplies to become one erosion polygon per time interval

# %%
if not EROSION_BY_DATE:    
    new_erosion_gdf = []

    def subtract_multiple_geoms(polygon, cutting_geometries):
            new_polygon = polygon
            for cutting_geom in cutting_geometries:
                new_polygon = new_polygon.difference(cutting_geom)
            return new_polygon

    for ddem in demcollection.ddems:
        relevant_glacier_polygons = glaciers_gdf.loc[glaciers_gdf.date.isin([ddem.interval.left, ddem.interval.right])]
        print(f"Cropping with {len(relevant_glacier_polygons)} glacier polygons.")
        differenced_geoms = erosion_vector.ds.geometry.apply(
            lambda geom: subtract_multiple_geoms(geom, relevant_glacier_polygons.geometry)
        )
        new_erosion_gdf.append(
            gpd.GeoDataFrame(
                {
                    'geometry': differenced_geoms,
                    'type': erosion_vector.ds['type'],
                    'interval': np.full(
                        len(differenced_geoms),
                        ddem.interval
                    )
                }
            )
        )
    ## also do it for bounding dataset
    relevant_glacier_polygons = glaciers_gdf.loc[glaciers_gdf.date.isin([demcollection.ddems[0].interval.left, demcollection.ddems[-1].interval.right])]
    differenced_geoms = erosion_vector.ds.geometry.apply(
        lambda geom: subtract_multiple_geoms(geom, relevant_glacier_polygons.geometry)
    )
    new_erosion_gdf.append(
            gpd.GeoDataFrame(
                {
                    'geometry': differenced_geoms,
                    'type': erosion_vector.ds['type'],
                    'interval': np.full(
                        len(differenced_geoms),
                        pd.Interval(demcollection.ddems[0].interval.left, demcollection.ddems[-1].interval.right)
                    )
                }
            )
        )
    

    ## also do it for the bounding dataset 
    relevant_glacier_polygons = glaciers_gdf.loc[glaciers_gdf.date.isin([demcollection.ddems[0].interval.left, demcollection.ddems[-1].interval.right])]
    differenced_geoms = erosion_vector.ds.geometry.apply(
        lambda geom: subtract_multiple_geoms(geom, relevant_glacier_polygons.geometry)
    )
    new_erosion_gdf.append(
            gpd.GeoDataFrame(
                {
                    'geometry': differenced_geoms,
                    'type': erosion_vector.ds['type'],
                    'interval': np.full(
                        len(differenced_geoms),
                        pd.Interval(demcollection.ddems[0].interval.left, demcollection.ddems[-1].interval.right)
                    )
                }
            )
        )

    
    erosion_vector.ds = new_erosion_gdf = pd.concat(new_erosion_gdf)

    src = new_erosion_gdf.copy()
    src['interval'] = src['interval'].apply(lambda x: x.left.strftime(strip_time_format))
    src.to_file(erosion_polygons_cropped_by_glaciers_output_file, driver='GeoJSON')

# %% [markdown]
# ### Split erosion vector into dictionary that organizes erosion polygons by a pd.Interval(start_date, end_Date)
#
# We do this so that DEMCollection.get_dv_series assigns the correct polygons to the correct dDEMs

# %%
if EROSION_BY_DATE:
    # need to create a column "interval" for sorting. Columns 'start_date' and 'end_date' should be in the erosion polygons file if `EROSION_BY_DATE`
    erosion_vector.ds['interval'] = erosion_vector.ds.apply(
        lambda row: pd.Interval(
            pd.Timestamp(datetime.strptime(row['start_date'], strip_time_format)),
            pd.Timestamp(datetime.strptime(row['end_date'], strip_time_format)),
        ), 
        axis=1
    )

start_date_to_gfd = dict(list(erosion_vector.ds.groupby("interval")))
start_date_to_gfd = dict({(key, gu.Vector(gdf)) for key, gdf in start_date_to_gfd.items()})
demcollection.outlines = start_date_to_gfd

# %% [markdown]
# Plot erosion geoms by date

# %%
grouped_erosion_vector_gdf = erosion_vector.ds.groupby('interval')
for tup in list(grouped_erosion_vector_gdf):
    interval = tup[0]
    gdf = tup[1]
    gdf.plot()
    plt.gca().set_title(str(interval))
    plt.show()

# %% [markdown]
# ## Plot DoDs

# %%
fig, axes = demcollection.plot_ddems(
    figsize=plot_figsize, vmin=plot_vmin, vmax=plot_vmax, 
    interpolation = "none", 
    plot_outlines=True,
    hillshade=True,
    cmap_alpha=0.15
)
plt.suptitle("dDEM Gallery, glacier signals removed")
fig.savefig(os.path.join(plot_output_dir, "dod_gallery_glaciers_masked.png"))
plt.show()

# %% [markdown]
# ## Mask Extra Signals

# %%
if MASK_EXTRA_SIGNALS:
    for ddem in demcollection.ddems:
        local_erosion_vector = erosion_vector.copy()
        local_erosion_vector.ds = local_erosion_vector.ds[local_erosion_vector.ds['interval'] == ddem.interval]
        extra_signals_mask = ~local_erosion_vector.create_mask(ddem).squeeze()
        ddem.data.mask = np.logical_or(ddem.data.mask, extra_signals_mask)

# %% [markdown]
# ## Plot DoDs

# %%
fig, axes = demcollection.plot_ddems(
    figsize=plot_figsize, vmin=plot_vmin, vmax=plot_vmax, 
    interpolation = "none", 
    plot_outlines=True,
    hillshade=True,
    cmap_alpha=0.15
)
plt.suptitle("dDEM Gallery, glacier signals and signals outside of study areas removed")
fig.savefig(os.path.join(plot_output_dir, "dod_gallery_glaciers_and_extra_masked.png"))
plt.show()

# %% [markdown]
# ## Interpolate

# %%
if INTERPOLATE:
    interpolated_ddems = demcollection.interpolate_ddems(max_search_distance=interpolation_max_search_distance)
    demcollection.set_ddem_filled_data()

# %% [markdown]
# ## Mask Extra Signals (again)
#
# We need to do this because we may have added some pixels buffered around the erosion polygons during the interpolation

# %%
if MASK_EXTRA_SIGNALS:
    for ddem in demcollection.ddems:
        local_erosion_vector = erosion_vector.copy()
        local_erosion_vector.ds = local_erosion_vector.ds[local_erosion_vector.ds['interval'] == ddem.interval]
        extra_signals_mask = ~local_erosion_vector.create_mask(ddem).squeeze()
        ddem.data.mask = np.logical_or(ddem.data.mask, extra_signals_mask)

# %% [markdown]
# ## Plot DoDs

# %%
fig, axes = demcollection.plot_ddems(
    figsize=plot_figsize, vmin=plot_vmin, vmax=plot_vmax, 
    interpolation = "none", 
    plot_outlines=True,
    hillshade=True,
    cmap_alpha=0.15
)
plt.suptitle("dDEM Gallery with interpolation, glacier signals and signals outside of study areas removed")
fig.savefig(os.path.join(plot_output_dir, "dods_final_interpolated.png"))
plt.show()

# %% [markdown]
# ## Plot distributions of data

# %%
fig, axes = plt.subplots(len(demcollection.ddems), figsize=(6,5), sharex=True, sharey=True)
for i, ddem in enumerate(demcollection.ddems):
    sns.distplot(ddem.data.filled(np.nan), ax=axes[i], hist=False)
    axes[i].set_ylabel("")
    axes[i].annotate(str(ddem.interval), xy=(5,0.4))
fig.text(-0.02, 0.5, 'common Y', va='center', rotation='vertical')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Save dDEMs to tif
#
# The datasets generated here are used to help creation of erosion polygons (and others)

# %%
if SAVE_DDEMS:
    # Save all interval dDEMs
    os.makedirs(dods_output_path, exist_ok=True)

    for ddem in demcollection.ddems:
        startt = ddem.start_time.strftime(strip_time_format)
        endt = ddem.end_time.strftime(strip_time_format)
        if RUN_LARGER_AREA:
            fn = f"{startt}_to_{endt}_largerarea.tif"
        else:
            fn = f"{startt}_to_{endt}.tif"
        fn = os.path.join(dods_output_path, fn)
        print(fn)
        ddem_xr = ddem.to_xarray()
        ddem_xr.data = ddem.data.filled(np.nan)
        ddem_xr.rio.to_raster(fn)

    # Save bounding dDEM

    bounding_ddem = xdem.dDEM(  
        demcollection.dems[-1] - demcollection.dems[0],
        demcollection.timestamps[0], 
        demcollection.timestamps[-1]
    )
    filled_data = bounding_ddem.interpolate(
        method="linear", 
        reference_elevation=demcollection.reference_dem, 
        max_search_distance=interpolation_max_search_distance
    )
    bounding_ddem.set_filled_data()

    # Mask out areas that are not within erosion vector. 
    # We only want to include areas that are within the erosion vectors for both dates.
    # We want to mask out areas that are not in the erosion polygons for the start date and not in the erosion polygons for the end date
    local_erosion_vector = erosion_vector.copy()
    #grab erosion polygons associated with the bounding interval 
    local_erosion_vector.ds = local_erosion_vector.ds[local_erosion_vector.ds['interval'] == bounding_ddem.interval]
    signal_we_want_mask = local_erosion_vector.create_mask(bounding_ddem).squeeze()
    bounding_ddem.data.mask = ~signal_we_want_mask
    bounding_ddem_xr = bounding_ddem.to_xarray()
    bounding_ddem_xr.data = bounding_ddem.data.filled(np.nan)
    startt = pd.Timestamp(bounding_ddem.start_time).strftime(strip_time_format)
    endt = pd.Timestamp(bounding_ddem.end_time).strftime(strip_time_format)
    if RUN_LARGER_AREA:
        fn = f"{startt}_to_{endt}_largerarea.tif"
    else:
        fn = f"{startt}_to_{endt}.tif"
    fn = os.path.join(dods_output_path, fn)
    print(fn)
    # bounding_ddem_copy = bounding_ddem.copy()
    bounding_ddem_xr.rio.to_raster(fn)

# %% [markdown]
# ## Mass wasting calculations

# %% [markdown]
# ## Create new datasets for our calculations

# %% [markdown]
# ### Define thresholding function
# * Make sure to set values equal to 0 instead of actually removing them!!

# %%
from scipy import stats
def threshold_ddem(ddem):
    ddem = ddem.copy()
    sample = ddem.data.compressed()
    datum = uncertainty_df.loc[uncertainty_df['Interval'] == ddem.interval]
    assert len(datum) == 1
    low = datum['90% CI Lower Bound'].iloc[0]
    hi = datum['90% CI Upper Bound'].iloc[0]
    print((low, hi))
    ddem.data[
        np.logical_and(ddem.data>low, ddem.data<hi)
    ] = 0
    
    return ddem


# %% [markdown]
# ### Create thresholded DEM collection

# %%
threshold_demcollection = xdem.DEMCollection(
    demcollection.dems,
    demcollection.timestamps
)

threshold_demcollection.ddems_are_intervalwise = True
threshold_demcollection.ddems = [threshold_ddem(ddem) for ddem in demcollection.ddems]
threshold_demcollection.outlines = demcollection.outlines


# %% [markdown]
# ### Create positive and negative DEM collections

# %%
def create_positive_and_negative_ddems(ddem):
    pos = ddem.copy()
    neg = ddem.copy()
    pos.data = np.ma.masked_less(pos.data, 0)
    neg.data = np.ma.masked_greater(neg.data, 0)
    return pos, neg

pos_ddems, neg_ddems = zip(*[create_positive_and_negative_ddems(ddem) for ddem in demcollection.ddems])

pos_ddemcollection = xdem.DEMCollection(
    demcollection.dems,
    demcollection.timestamps
)
pos_ddemcollection.ddems_are_intervalwise = True
pos_ddemcollection.ddems = pos_ddems
pos_ddemcollection.outlines = demcollection.outlines

neg_ddemcollection = xdem.DEMCollection(
    demcollection.dems,
    demcollection.timestamps
)
neg_ddemcollection.ddems_are_intervalwise = True
neg_ddemcollection.ddems = neg_ddems
neg_ddemcollection.outlines = demcollection.outlines

# %% [markdown]
# ### Create thresholded positive and negative DEM collections

# %%
threshold_pos_ddems, threshold_neg_ddems = zip(*[create_positive_and_negative_ddems(ddem) for ddem in threshold_demcollection.ddems])


threshold_pos_ddemcollection = xdem.DEMCollection(
    threshold_demcollection.dems,
    threshold_demcollection.timestamps
)
threshold_pos_ddemcollection.ddems_are_intervalwise = True
threshold_pos_ddemcollection.ddems = threshold_pos_ddems
threshold_pos_ddemcollection.outlines = threshold_demcollection.outlines

threshold_neg_ddemcollection = xdem.DEMCollection(
    threshold_demcollection.dems,
    threshold_demcollection.timestamps
)
threshold_neg_ddemcollection.ddems_are_intervalwise = True
threshold_neg_ddemcollection.ddems = threshold_neg_ddems
threshold_neg_ddemcollection.outlines = demcollection.outlines

# %% [markdown]
# ### Create bounding DEM collection

# %%
# Create bounding interval 
bounding_interval = pd.Interval(pd.Timestamp(demcollection.timestamps[0]), pd.Timestamp(demcollection.timestamps[-1]))
print(f"Bounding data based on times: {bounding_interval}")
# Get bounding outlines
bounding_outlines = demcollection.outlines.get(bounding_interval)
#Create bounding dem collection
bounding_dem_collection = xdem.DEMCollection(
    [demcollection.dems[0], demcollection.dems[-1]],
    [demcollection.timestamps[0], demcollection.timestamps[-1]],
    outlines = bounding_outlines
)

_ = bounding_dem_collection.subtract_dems_intervalwise()

# filter outliers
if FILTER_OUTLIERS:
    if SIMPLE_FILTER:
        for dh in bounding_dem_collection.ddems:
            dh.data = np.ma.masked_where(np.abs(dh.data) > simple_filter_threshold, dh.data)
    else:
        for dh in bounding_dem_collection.ddems:
            all_values_masked = dh.data.copy()
            all_values = all_values_masked.filled(np.nan)
            low = np.nanmedian(all_values) - 4*xdem.spatialstats.nmad(all_values)
            high = np.nanmedian(all_values) + 4*xdem.spatialstats.nmad(all_values)
            print(np.nanmax(dh.data))
            print(np.nanmin(dh.data))
            print(dh.interval)
            print(low)
            print(high)
            all_values_masked = np.ma.masked_greater(all_values_masked, high)
            all_values_masked = np.ma.masked_less(all_values_masked, low)
            dh.data = all_values_masked
            print(np.nanmax(dh.data))
            print(np.nanmin(dh.data))
            print()

# interpolate
if INTERPOLATE:
    interpolated_ddems = bounding_dem_collection.interpolate_ddems(max_search_distance=interpolation_max_search_distance)
    bounding_dem_collection.set_ddem_filled_data()

# %% [markdown]
# ## Calculations

# %% [markdown]
# ### Net mass wasted

# %%
dv_df = demcollection.get_dv_series(return_area=True).reset_index()
dv_df.head()

# %%
bounding_dv_df = bounding_dem_collection.get_dv_series(return_area=True).reset_index()
bounding_dv_df.head()

# %% [markdown]
# ### Net mass wasted by erosion type

# %%
hillslope_dv_df = demcollection.get_dv_series(return_area=True, outlines_filter="type == 'hillslope'").reset_index()
hillslope_dv_df['type'] = 'hillslope'
fluvial_dv_df = demcollection.get_dv_series(return_area=True, outlines_filter="type == 'fluvial'").reset_index()
fluvial_dv_df['type'] = 'fluvial'

# %% [markdown]
# ### Gross positive and negative mass wasted

# %%
pos_dv_df = pos_ddemcollection.get_dv_series(return_area=True).reset_index()
neg_dv_df = neg_ddemcollection.get_dv_series(return_area=True).reset_index()

# %% [markdown]
# ### Gross positive and negative mass wasted, by erosion type

# %%
hillslope_pos_dv_df = pos_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'hillslope'").reset_index()
hillslope_pos_dv_df['type'] = 'hillslope'

fluvial_pos_dv_df = pos_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'fluvial'").reset_index()
fluvial_pos_dv_df['type'] = 'fluvial'

hillslope_neg_dv_df = neg_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'hillslope'").reset_index()
hillslope_neg_dv_df['type'] = 'hillslope'

fluvial_neg_dv_df = neg_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'fluvial'").reset_index()
fluvial_neg_dv_df['type'] = 'fluvial'

# %% [markdown]
# ### Gross positive and negative mass wasted with threshold (1 meter)

# %%
threshold_pos_dv_df = threshold_pos_ddemcollection.get_dv_series(return_area=True).reset_index()
threshold_neg_dv_df = threshold_neg_ddemcollection.get_dv_series(return_area=True).reset_index()

# %% [markdown]
# ### Gross positive and negative mass wasted with threshold (1 meter), by erosion type

# %%
hillslope_threshold_pos_dv_df = threshold_pos_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'hillslope'").reset_index()
hillslope_threshold_pos_dv_df['type'] = 'hillslope'

fluvial_threshold_pos_dv_df = threshold_pos_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'fluvial'").reset_index()
fluvial_threshold_pos_dv_df['type'] = 'fluvial'

hillslope_threshold_neg_dv_df = threshold_neg_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'hillslope'").reset_index()
hillslope_threshold_neg_dv_df['type'] = 'hillslope'

fluvial_threshold_neg_dv_df = threshold_neg_ddemcollection.get_dv_series(return_area=True, outlines_filter="type == 'fluvial'").reset_index()
fluvial_threshold_neg_dv_df['type'] = 'fluvial'


# %% [markdown]
# ### Add metadata to all the dataframes resulting from the calculations
#
# Maybe this should be added as functionality to DEMCollection?

# %%
def enrich_volume_data(df, pixel_area, pixel_side_length, uncertainty_df):
    """Modify the resulting dataframe of `demcollection.get_dv_series` by 
    adding a bunch of useful data. Calculates volumetric uncertainty as well.

    Args:
        df (_type_): _description_
        pixel_area (_type_): _description_
    """
    df["n_pixels"] = df["area"]/pixel_area

    df["volumetric_uncertainty"] = df.apply(
        lambda row: xdem.spatialstats.volumetric_uncertainty(
            n_pixels = row["n_pixels"],
            pixel_side_length = pixel_side_length,
            rmse = uncertainty_df.loc[uncertainty_df['Interval'] == row['index']]['RMSE'].iloc[0],
            mean = uncertainty_df.loc[uncertainty_df['Interval'] == row['index']]['Mean'].iloc[0],
            range_val = uncertainty_df.loc[uncertainty_df['Interval'] == row['index']]['Range'].iloc[0],
            sill_val = uncertainty_df.loc[uncertainty_df['Interval'] == row['index']]['Sill'].iloc[0],
        ),
        axis=1
    )
    df['start_time'] = df['index'].apply(lambda x: x.left)
    df['end_time'] = df['index'].apply(lambda x: x.right)
    df['time_difference_years'] = df.apply(
        lambda row: round((row['end_time'] - row['start_time']).days/365.25),
        axis=1
    )
    df['Annual Mass Wasted'] = df['volume']/df['time_difference_years']
    #### #### #### #### #### #### #### #### #### #### #### #### 
    #### 
    #### ToDo: Confirm this is the proper calculation:
    #### 
    #### #### #### #### #### #### #### #### #### #### #### #### 
    df["Upper CI"] = (df['volume'] + df['volumetric_uncertainty'])/df['time_difference_years']
    df["Lower CI"] = (df['volume'] - df['volumetric_uncertainty'])/df['time_difference_years']
    df["Average Date"] = df['start_time'] + ((df['end_time'] - df['start_time']) / 2).dt.ceil('D')
    return df


# %%
demcollection.reference_dem.res[0], demcollection.reference_dem.res[1]

# %%
bounding_dv_df

# %%
uncertainty_df

# %%
dv_df = enrich_volume_data(
    dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

bounding_dv_df = enrich_volume_data(
    bounding_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

# %%


fluvial_dv_df = enrich_volume_data(
    fluvial_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

hillslope_dv_df = enrich_volume_data(
    hillslope_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

pos_dv_df = enrich_volume_data(
    pos_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

neg_dv_df = enrich_volume_data(
    neg_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

hillslope_pos_dv_df = enrich_volume_data(
    hillslope_pos_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)
fluvial_pos_dv_df = enrich_volume_data(
    fluvial_pos_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)
hillslope_neg_dv_df = enrich_volume_data(
    hillslope_neg_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)
fluvial_neg_dv_df = enrich_volume_data(
    fluvial_neg_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0], 
    uncertainty_df = uncertainty_df
)

threshold_pos_dv_df = enrich_volume_data(
    threshold_pos_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0],
    uncertainty_df = uncertainty_df
)
threshold_neg_dv_df = enrich_volume_data(
    threshold_neg_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0],
    uncertainty_df = uncertainty_df
)
hillslope_threshold_pos_dv_df = enrich_volume_data(
    hillslope_threshold_pos_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0],
    uncertainty_df = uncertainty_df
)
fluvial_threshold_pos_dv_df = enrich_volume_data(
    fluvial_threshold_pos_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0],
    uncertainty_df = uncertainty_df
)
hillslope_threshold_neg_dv_df = enrich_volume_data(
    hillslope_threshold_neg_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0],
    uncertainty_df = uncertainty_df
)
fluvial_threshold_neg_dv_df = enrich_volume_data(
    fluvial_threshold_neg_dv_df,
    pixel_area = demcollection.reference_dem.res[0] * demcollection.reference_dem.res[1],
    pixel_side_length = demcollection.reference_dem.res[0],
    uncertainty_df = uncertainty_df
)

# %% [markdown]
# # Plot

# %% [markdown]
# #### Plot net mass wasted

# %%
bars = alt.Chart(dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
    opacity=0.8
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", 
    title="Annualized rate of volumetric change, in m³/yr"
    )
).properties(
    # width=300, 
    # height=300
)

characters = '⁰ ¹ ² ³ ⁴ ⁵'
error_bars = alt.Chart(dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", axis=alt.Axis(format=("%Y")), title=""),
    alt.Y("Lower CI"),
    alt.Y2("Upper CI")
)

chart = bars + error_bars

# chart.save(os.path.join(plot_output_dir, "mass_wasted_net.png"), scale_factor=2.0)

chart.properties(
    title="Net volume change over dDEM intervals"
)

# %%
cum_dv_df = dv_df.copy()
cum_dv_df['cumulative volume'] = dv_df['volume'].cumsum()
cum_dv_df['Lower CI'] = 0
cum_dv_df['Upper CI'] = 0
cum_dv_df.loc[len(cum_dv_df) - 1, 'Lower CI'] = cum_dv_df.loc[len(cum_dv_df) - 1, 'cumulative volume'] - np.sqrt(
    (cum_dv_df['volumetric_uncertainty']**2).sum()
)
cum_dv_df.loc[len(cum_dv_df) - 1, 'Upper CI'] = cum_dv_df.loc[len(cum_dv_df) - 1, 'cumulative volume'] + np.sqrt(
    (cum_dv_df['volumetric_uncertainty']**2).sum()
)
cum_dv_df = pd.concat([
        cum_dv_df,
        pd.DataFrame([{
            'cumulative volume': 0,
            'end_time': cum_dv_df.iloc[0]['start_time'],
            'volumetric_uncertainty': 0
        }])
    ],
    ignore_index=True
)
cum_dv_df['end_time'] = cum_dv_df['end_time'].apply(pd.Timestamp)
cum_dv_df.head(2)

# %%
bounding_cum_dv_df = bounding_dv_df.copy()
bounding_cum_dv_df['cumulative volume'] = bounding_dv_df['volume'].cumsum()
bounding_cum_dv_df['Lower CI'] = 0
bounding_cum_dv_df['Upper CI'] = 0
bounding_cum_dv_df.loc[len(bounding_cum_dv_df) - 1, 'Lower CI'] = bounding_cum_dv_df.loc[len(bounding_cum_dv_df) - 1, 'cumulative volume'] - np.sqrt(
    (bounding_cum_dv_df['volumetric_uncertainty']**2).sum()
)
bounding_cum_dv_df.loc[len(bounding_cum_dv_df) - 1, 'Upper CI'] = bounding_cum_dv_df.loc[len(bounding_cum_dv_df) - 1, 'cumulative volume'] + np.sqrt(
    (bounding_cum_dv_df['volumetric_uncertainty']**2).sum()
)
bounding_cum_dv_df = pd.concat([
        bounding_cum_dv_df,
        pd.DataFrame([{
            'cumulative volume': 0,
            'end_time': bounding_cum_dv_df.iloc[0]['start_time'],
            'volumetric_uncertainty': 0
        }])
    ],
    ignore_index=True
)

bounding_cum_dv_df['end_time'] = bounding_cum_dv_df['end_time'].apply(pd.Timestamp)
bounding_cum_dv_df.head(2)

# %%
cum_plot = alt.Chart(cum_dv_df.drop(columns='index')).mark_line(point=True).encode(
    alt.X('end_time:T', title='Time'),
    alt.Y('cumulative volume:Q', 
    title='Cumulative net change, in m³/yr')
)

error_bars = alt.Chart(cum_dv_df.drop(columns="index")).mark_bar(
    width=1
).encode(
    alt.X("end_time:T"),
    alt.Y("Lower CI"),
    alt.Y2("Upper CI")
)

bounding_point = alt.Chart(bounding_cum_dv_df.drop(columns='index')).mark_circle(shape='diamond', color='red', size=100).encode(
    alt.X("end_time:T"),
    alt.Y("volume:Q"),
)

bounding_point_error_bars = alt.Chart(bounding_cum_dv_df.drop(columns='index')).mark_bar(
    color="red",
    width=1
).encode(
    alt.X("end_time:T", title=""),
    alt.Y("Lower CI"),
    alt.Y2("Upper CI")
)

(bounding_point + bounding_point_error_bars + error_bars + cum_plot).properties(
    height=200,
    title="Cumulative volume change over dDEM intervals",
)

# %%
dods_output_path, plot_output_dir

# %% [markdown]
# #### Plot cumulative net mass wasted

# %% [markdown]
# #### Plot net mass wasted by erosion type

# %%
bars = alt.Chart(fluvial_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
    opacity=0.8
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    width=300, 
    height=150
)

error_bars = alt.Chart(fluvial_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
).properties(
    width=300, 
    height=150
)

fluvial_chart = bars + error_bars

bars = alt.Chart(hillslope_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
    opacity=0.8
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    width=300, 
    height=150
)

error_bars = alt.Chart(hillslope_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
).properties(
    width=300, 
    height=150
)

hillslope_chart = bars + error_bars

# chart.save(os.path.join(plot_output_dir, "mass_wasted_net.png"), scale_factor=2.0)

(fluvial_chart.properties(title='fluvial') & hillslope_chart.properties(title='hillslope')).resolve_scale(x='shared').properties(
    title="Net volume change over dDEM intervals, split by erosion type",
)

# %% [markdown]
# #### Plot gross positive and negative mass wasted

# %%
bars_neg = alt.Chart(neg_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
    color="red"
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_neg = alt.Chart(neg_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

bars_pos = alt.Chart(pos_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_pos = alt.Chart(pos_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

chart = (bars_pos + error_bars_pos + bars_neg + error_bars_neg)
# chart.save(os.path.join(plot_output_dir, "mass_wasted_gross.png"), scale_factor=2.0)
chart.properties(
    title={
        'text': ["Gross positive and negative volume changes over dDEM intervals"],
    }
)

# %% [markdown]
# ####  Plot gross positive and negative mass wasted, thresholded

# %%
threshold_pos_dv_df['Annual Mass Wasted'],pos_dv_df['Annual Mass Wasted'],

# %%
bars_neg = alt.Chart(threshold_neg_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
    color="red"
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_neg = alt.Chart(threshold_neg_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

bars_pos = alt.Chart(threshold_pos_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_pos = alt.Chart(threshold_pos_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

chart = (bars_pos + error_bars_pos + bars_neg + error_bars_neg)
# chart.save(os.path.join(plot_output_dir, "mass_wasted_gross.png"), scale_factor=2.0)

chart = chart.properties(
    title={
        'text': ["Gross positive and negative volume changes over dDEM intervals"],
        'subtitle': [f"Threshold of 90% CI applied"]
    }
)
chart

# %% [markdown]
# #### Plot gross positive and negative mass wasted, by erosion type

# %%
bars_neg = alt.Chart(fluvial_neg_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
    color="red"
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_neg = alt.Chart(fluvial_neg_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

bars_pos = alt.Chart(fluvial_pos_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_pos = alt.Chart(fluvial_pos_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

chart_fluvial = (bars_pos + error_bars_pos + bars_neg + error_bars_neg).properties(title='fluvial')

# %%
bars_neg = alt.Chart(hillslope_neg_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
    color="red"
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_neg = alt.Chart(hillslope_neg_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

bars_pos = alt.Chart(hillslope_pos_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_pos = alt.Chart(hillslope_pos_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

chart_hillslope = (bars_pos + error_bars_pos + bars_neg + error_bars_neg).properties(title='hillslope')

# %%
(chart_fluvial & chart_hillslope).properties(
    title={
        'text': ["Gross positive and negative volume changes over dDEM intervals, split by erosion type"]    }
)

# %% [markdown]
# #### Plot  gros positive and negative mass wasted by erosion type, thresholded

# %%
bars_neg = alt.Chart(fluvial_threshold_neg_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
    color="red"
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_neg = alt.Chart(fluvial_threshold_neg_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

bars_pos = alt.Chart(fluvial_threshold_pos_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 3,
    stroke="white",
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_pos = alt.Chart(fluvial_threshold_pos_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

chart_fluvial = (bars_pos + error_bars_pos + bars_neg + error_bars_neg).properties(title='fluvial')

# %%
bars_neg = alt.Chart(hillslope_threshold_neg_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
    color="red"
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_neg = alt.Chart(hillslope_threshold_neg_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

bars_pos = alt.Chart(hillslope_threshold_pos_dv_df.drop(columns='index')).mark_bar(
    strokeWidth = 1.5,
    stroke="white",
).encode(
    alt.X('start_time:T'),
    alt.X2('end_time:T'),
    alt.Y("Annual Mass Wasted:Q", title="Annualized rate of volumetric change, in m³/yr"),
).properties(
    # width=300, 
    # height=300
)

error_bars_pos = alt.Chart(hillslope_threshold_pos_dv_df.drop(columns="index")).mark_bar(
    color="black",
    width=2
).encode(
    alt.X("Average Date:T", title=""),
    alt.Y("Lower CI", title="Annualized rate of volumetric change, in m³/yr"),
    alt.Y2("Upper CI")
)

chart_hillslope = (bars_pos + error_bars_pos + bars_neg + error_bars_neg).properties(title='hillslope')

# %%
chart = (chart_fluvial & chart_hillslope).properties(
    title={
        'text': ["Gross positive and negative volume changes over dDEM intervals, split by erosion type"],
        'subtitle': [f"Threshold of 90% CI applied"]
    }
)
chart

# %% [markdown]
# ## Save dataframes

# %%
dfs = [
    cum_dv_df,
    bounding_cum_dv_df,

    dv_df,
    bounding_dv_df,
    # fluvial_dv_df,
    # hillslope_dv_df,
    # threshold_dv_df,
    # hillslope_threshold_dv_df,
    # fluvial_threshold_dv_df,
    
    # pos_dv_df,
    # neg_dv_df,
    # hillslope_pos_dv_df,
    # fluvial_pos_dv_df,
    # hillslope_neg_dv_df,
    # fluvial_neg_dv_df,
    
    threshold_pos_dv_df,
    threshold_neg_dv_df,
    hillslope_threshold_pos_dv_df,
    fluvial_threshold_pos_dv_df,
    hillslope_threshold_neg_dv_df,
    fluvial_threshold_neg_dv_df,
]

names = [
    'cum_dv_df',
    'bounding_cum_dv_df',

    'dv_df',
    'bounding_dv_df',
    # 'fluvial_dv_df',
    # 'hillslope_dv_df',
    # 'threshold_dv_df',
    # 'hillslope_threshold_dv_df',
    # 'fluvial_threshold_dv_df',
    
    # 'pos_dv_df',
    # 'neg_dv_df',
    # 'hillslope_pos_dv_df',
    # 'fluvial_pos_dv_df',
    # 'hillslope_neg_dv_df',
    # 'fluvial_neg_dv_df',
    'threshold_pos_dv_df',
    'threshold_neg_dv_df',
    'hillslope_threshold_pos_dv_df',
    'fluvial_threshold_pos_dv_df',
    'hillslope_threshold_neg_dv_df',
    'fluvial_threshold_neg_dv_df',
]
for df,name in zip(dfs, names):
    df['valley'] = valley_name
    if RUN_LARGER_AREA:
        outdir = os.path.join("outputs", "larger_area", name)
    else:
        outdir = os.path.join("outputs", name)
    outfile = os.path.join(outdir, valley_name + ".pickle")
    os.makedirs(outdir, exist_ok=True)
    print(outfile)
    df.to_pickle(outfile)
# %%

