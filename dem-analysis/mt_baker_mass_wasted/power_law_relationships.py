# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3.9.2 ('xdem')
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
from shapely import wkt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

from functools import reduce
import scipy

np.set_printoptions(suppress=True)
from sklearn.metrics import r2_score


BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
print(f"retrieved base path: {BASE_PATH}")


# %%
def nse(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(predictions))**2))


# %%
def r2_plot(df, x, y, plot_fit_line=False, r2_annotations=False, limit=150000, x_axis_title='', y_axis_title='', size=100):
    chart = alt.Chart(df).mark_circle(size=size).encode(
        x=alt.X(x, title=x_axis_title),
        y=alt.Y(y, title=y_axis_title)
    )
    line = chart.transform_regression(x, y).mark_line()

    params = alt.Chart(df).transform_regression(
        x, y, params=True
    ).mark_text(align='left').encode(
        x=alt.value(20),  # pixels from left
        y=alt.value(20),  # pixels from top
        text=alt.Text('rSquared:N', format=".3f")
    ).properties(width=200, height=200)

    one_to_one_line = alt.Chart(pd.DataFrame({
        'x': np.linspace(0, limit, 100),
        'y': np.linspace(0, limit, 100)
    })).mark_line(color='black', opacity=0.25).encode(
        alt.X('x', scale=alt.Scale(domain=(0,limit))),
        alt.Y('y', scale=alt.Scale(domain=(0,limit)))
    )

    if r2_annotations:
        base_chart = (chart + params + one_to_one_line).properties(width=200, height=200)
    else:
        base_chart = (chart + one_to_one_line).properties(width=200, height=200)
    if plot_fit_line:
        return base_chart + line
    else:
        return base_chart


# %% [markdown]
# # Set constants

# %%
porosity = 0.35
density_kg_per_cubic_meter = 2600
kg_per_metric_ton = 1000

# %% [markdown]
# # Load Data

# %% [markdown]
# ### Load streamstats wastersheds

# %%
streamstats_watersheds_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/streamstats_watersheds/*.geojson"))

streamstats_gdf = gpd.GeoDataFrame()
for f in streamstats_watersheds_fns:
    new_data = gpd.read_file(f)
    new_data['Valley Name'] = f.split("/")[-1].split(".geojson")[0]
    streamstats_gdf = pd.concat([streamstats_gdf, new_data])
streamstats_gdf['Valley Name'] = streamstats_gdf['Valley Name'].str.title()
streamstats_gdf = streamstats_gdf[streamstats_gdf.geometry.type == 'Polygon']
streamstats_gdf = streamstats_gdf.to_crs("EPSG:32610")
streamstats_gdf['watershed area (square m)'] = streamstats_gdf.geometry.area
streamstats_gdf['watershed area (square km)'] = streamstats_gdf['watershed area (square m)'] / 10**6

# %%
streamstats_gdf

# %% [markdown]
# ### Load glacier change measurements

# %%
glacier_change_df = pd.read_pickle("outputs/glacier_area.pickle").reset_index()

# %%
glacier_change_df = pd.read_pickle("outputs/glacier_area.pickle").reset_index()
glacier_change_df['Valley Name'] = glacier_change_df['Name'].apply(lambda s: s.split(" ")[0])
glacier_change_df['glacial advance absolute'] = glacier_change_df.apply(
    lambda row: row['area difference']['1977_09_27'] 
        if not np.isnan(row['area difference']['1977_09_27'])
        else row['area difference']['1979_10_06'],
    axis=1
)
glacier_change_df['glacial retreat absolute'] = glacier_change_df['area difference']['2015_09_01']
glacier_change_df['glacial advance and retreat absolute'] = np.abs(glacier_change_df['glacial advance absolute']) + np.abs(glacier_change_df['glacial retreat absolute'])
glacier_change_df['glacial area 1947'] = glacier_change_df['area']['1947_09_14']
glacier_change_df['glacial area 1977/79'] = glacier_change_df['area'].apply(lambda row: row['1977_09_27'] if np.isnan(row['1979_10_06']) else row['1979_10_06'], axis=1)
glacier_change_df['glacial area 2015'] = glacier_change_df['area']['2015_09_01']

glacier_change_df = glacier_change_df[[
    'Valley Name',
    'glacial advance absolute',
    'glacial retreat absolute',
    'glacial advance and retreat absolute',
    'glacial area 1947',
    'glacial area 1977/79',
    'glacial area 2015'
]]

glacier_change_df.columns = glacier_change_df.columns.get_level_values(0)

# Manually combine the Coleman Roosevelt rows, remove the old ones, and add a new combined "Coleman" row
row = glacier_change_df[glacier_change_df['Valley Name'].isin(['Coleman', 'Roosevelt'])].sum()
row['Valley Name'] = 'Coleman'
glacier_change_df = pd.concat([
        glacier_change_df[~glacier_change_df['Valley Name'].isin(['Coleman', 'Roosevelt'])],
        row
    ], 
    ignore_index=True
)

# %% [markdown]
# ### Load terrain attribute data

# %%
terrain_attrs_erosionarea = pd.read_csv("outputs/terrain_attributes_erosionarea.csv")
terrain_attrs_erosionarea = terrain_attrs_erosionarea.rename(columns={'name': 'Valley Name'})
terrain_attrs_erosionarea['drainage area (km)'] = terrain_attrs_erosionarea['drainage area'] / 1e6


terrain_attrs_erosionarea_bytpe = pd.read_csv("outputs/terrain_attributes_erosionarea_bytype.csv")
terrain_attrs_erosionarea_bytpe = terrain_attrs_erosionarea_bytpe.rename(columns={'name': 'Valley Name'})
terrain_attrs_erosionarea_bytpe['drainage area (km)'] = terrain_attrs_erosionarea_bytpe['drainage area'] / 1e6

# %%
terrain_attrs_erosionarea[['slope all erosion area', 'drainage area (km)', 'curvature']] = terrain_attrs_erosionarea[['slope', 'drainage area (km)', 'curvature']]

terrain_attrs_erosionarea = terrain_attrs_erosionarea[['Valley Name', 'drainage area (km)', 'curvature', 'slope all erosion area']]

# %%
terrain_attrs_erosionarea = terrain_attrs_erosionarea.merge(
    terrain_attrs_erosionarea_bytpe.query("type == 'fluvial'")[['Valley Name', 'slope']].rename(columns={'slope': 'slope fluvial erosion area'}),
    on='Valley Name'
)

terrain_attrs_erosionarea = terrain_attrs_erosionarea.merge(
    terrain_attrs_erosionarea_bytpe.query("type == 'hillslope'")[['Valley Name', 'slope']].rename(columns={'slope': 'slope hillslope erosion area'}),
    on='Valley Name'
)

# %%
terrain_attrs_erosionarea

# %%
# ### Load longitudinal slope measurements
long_slopes_df = pd.read_csv(os.path.join(BASE_PATH, "hsfm-geomorph/data/slopes.csv"))
long_slopes_df

# %% [markdown]
# ### Load lithology data

# %%
lithology_df = pd.read_csv("outputs/lithology.csv")

# %%
lithology_df['nonigneous fraction'] = lithology_df['AREA']
lithology_df = lithology_df[['Valley Name', 'nonigneous fraction']]

# %%
lithology_df

# %% [markdown]
# ## Load gross volume change measurements

# %% [markdown]
# ### Load net volume change measurements

# %%
net_measurements = pd.read_pickle("outputs/xdem_whole_mountain_combined/dv_df_by_valley.pickle")
neg_measurements = pd.read_pickle("outputs/xdem_whole_mountain_combined/thresh_neg_dv_df_by_valley.pickle")
pos_measurements = pd.read_pickle("outputs/xdem_whole_mountain_combined/thresh_pos_dv_df_by_valley.pickle")

# %%
for df in [net_measurements, neg_measurements, pos_measurements]:
    df['Valley Name'] = df['name']
    df['time interval'] = df['index']

    df = df[[
        'time interval',
        'Valley Name',
        'dh',
        'area',
        'volume',
        'bounding',
        'n_pixels',
        'start_time',
        'end_time',
        'time_difference_years',
        'Annual Mass Wasted',
        'volumetric_uncertainty',
        'Upper CI',
        'Lower CI',
        'Average Date',   
    ]].reset_index()

# %% [markdown]
# #### Replace numbers for the Intensive Observation Areas with the case study calculations (more accurate)

# %%
pd.concat([pd.read_pickle(f) for f in glob.glob("outputs/larger_area/threshold_pos_dv_df/*.pickle")]).dropna(subset='Annual Mass Wasted')

# %%
# ls -lah outputs/larger_area/bounding_dv_df/

# %%
casestudy_netmeasurements_bounding = pd.concat([pd.read_pickle(f) for f in glob.glob("outputs/larger_area/bounding_dv_df/*.pickle")]).dropna(subset='Annual Mass Wasted')
# casestudy_negmeasurements_bounding = pd.concat([pd.read_pickle(f) for f in glob.glob("outputs/larger_area/threshold_neg_dv_df/*.pickle")]).dropna(subset='Annual Mass Wasted')
# casestudy_posmeasurements_bounding = pd.concat([pd.read_pickle(f) for f in glob.glob("outputs/larger_area/threshold_pos_dv_df/*.pickle")]).dropna(subset='Annual Mass Wasted')

casestudy_netmeasurements_bounding['bounding'] = True
casestudy_netmeasurements = pd.concat([pd.read_pickle(f) for f in glob.glob("outputs/larger_area/dv_df/*.pickle")]).dropna(subset='Annual Mass Wasted')
casestudy_netmeasurements['bounding'] = False
casestudy_netmeasurements = pd.concat([casestudy_netmeasurements, casestudy_netmeasurements_bounding])

# for df in []
casestudy_netmeasurements['Valley Name'] = casestudy_netmeasurements['valley']
casestudy_netmeasurements['time interval'] = casestudy_netmeasurements['index']
casestudy_netmeasurements = casestudy_netmeasurements[[
    'time interval',
    'Valley Name',
    'dh',
    'area',
    'volume',
    'bounding',
    'n_pixels',
    'start_time',
    'end_time',
    'time_difference_years',
    'Annual Mass Wasted',
    'volumetric_uncertainty',
    'Upper CI',
    'Lower CI',
    'Average Date',   
]].reset_index()

# %%
net_measurements = pd.concat([
    net_measurements[~net_measurements['Valley Name'].isin(['Deming', 'Mazama', 'Coleman',' Rainbow'])],
    casestudy_netmeasurements[casestudy_netmeasurements['Valley Name'].isin(['Deming', 'Mazama', 'Coleman',' Rainbow'])]
])

# %% [markdown]
# #### Name time intervals

# %%
date_interval_to_named_interval = {
    pd.Interval(pd.Timestamp(1947,9,14), pd.Timestamp(1977,9,27)): 'advance',
    pd.Interval(pd.Timestamp(1947,9,14), pd.Timestamp(1979,10,6)): 'advance',

    pd.Interval(pd.Timestamp(1977,9,27), pd.Timestamp(2015,9,1)): 'retreat',
    pd.Interval(pd.Timestamp(1979,10,6), pd.Timestamp(2015,9,1)): 'retreat',

    pd.Interval(pd.Timestamp(1947,9,14), pd.Timestamp(2015,9,1)): 'bounding'
}


net_measurements['named interval'] = net_measurements['time interval'].apply(date_interval_to_named_interval.get)

# %%
len(net_measurements['Valley Name'].unique())

# %% [markdown]
# ## Merge Datasets

# %% [markdown]
# ### Merge volume measurements and Streamstats watershed area data

# %%
net_measurements = net_measurements.merge(
    streamstats_gdf[['Valley Name', 'watershed area (square m)',	'watershed area (square km)']].reset_index(drop=True),
    on = 'Valley Name'
)

# %%
len(net_measurements['Valley Name'].unique())

# %% [markdown]
# ### Merge in Terrain Attributes data (attributes of erosion polygon area)

# %%
net_measurements = net_measurements.merge(
    terrain_attrs_erosionarea,
    on='Valley Name'
)

# %%
len(net_measurements['Valley Name'].unique())

# %% [markdown]
# ### Merge in glacier change data

# %%
net_measurements = net_measurements.merge(
    glacier_change_df, on='Valley Name'
)

# %%
len(net_measurements['Valley Name'].unique())

# %% [markdown]
# ### Merge in lithology data

# %%
net_measurements.head(3)

# %%
net_measurements = net_measurements.merge(
    lithology_df, on='Valley Name'
)

# %% [markdown]
# ##### Assign Nooksack river fork

# %%
net_measurements['Fork of the Nooksack River'] = 'Does not drain to Nooksack River'
net_measurements.loc[net_measurements["Valley Name"] == "Coleman", 'Fork of the Nooksack River'] = "North Fork"
net_measurements.loc[net_measurements["Valley Name"] == "Deming", 'Fork of the Nooksack River'] = "Middle Fork"
net_measurements.loc[net_measurements["Valley Name"] == "Mazama", 'Fork of the Nooksack River'] = "North Fork"
net_measurements.loc[net_measurements["Valley Name"] == "Thunder", 'Fork of the Nooksack River'] = "Middle Fork"

# %%
len(net_measurements['Valley Name'].unique())

# %% [markdown]
# ## Calculate sediment yields from volume measurements

# %%
net_measurements['sediment yield (t / yr)'] = - net_measurements['Annual Mass Wasted'] * (1 - porosity) * density_kg_per_cubic_meter / kg_per_metric_ton
net_measurements['Upper CI sediment yield'] = - net_measurements['Upper CI'] * (1 - porosity) * density_kg_per_cubic_meter / kg_per_metric_ton
net_measurements['Lower CI sediment yield'] = - net_measurements['Lower CI'] * (1 - porosity) * density_kg_per_cubic_meter / kg_per_metric_ton

net_measurements['sediment yield normalized (t / km^2 / yr)'] = net_measurements['sediment yield (t / yr)'] / net_measurements['watershed area (square km)']
net_measurements['Upper CI sediment yield normalized'] = net_measurements['Upper CI sediment yield'] / net_measurements['watershed area (square km)']
net_measurements['Lower CI sediment yield normalized'] = net_measurements['Lower CI sediment yield'] / net_measurements['watershed area (square km)']

net_measurements['Annual Mass Wasted normalized'] = net_measurements['Annual Mass Wasted'] / (net_measurements['watershed area (square km)']*(1000**2))
net_measurements['Upper CI normalized'] = net_measurements['Upper CI'] / (net_measurements['watershed area (square km)']*(1000**2))
net_measurements['Lower CI normalized'] = net_measurements['Lower CI'] / (net_measurements['watershed area (square km)']*(1000**2))

# %% [markdown]
# # Plot Sediment Yield (1947-2015)

# %%
yield_domain = [-50, 168.99999999999997]
volume_domain = [-29585.79881656805/1000, 100]

# %%
valley_sorting = ['Coleman','Deming','Rainbow','Mazama','Park','Easton','Boulder','Thunder','Squak','Talum']

# %%
fig_width=200
fig_height=300
yield_domain = [-11.8310517529, 8.4507512520868]
volume_domain = [-0.007, 0.005]

src_bounding = net_measurements[net_measurements['named interval'] == 'bounding'].drop(columns=['time interval']).drop(columns='index').drop(columns=[0])

src_bounding['Annual Mass Wasted normalized'] = -src_bounding['Annual Mass Wasted normalized']
src_bounding['Lower CI normalized'] = -src_bounding['Lower CI normalized']
src_bounding['Upper CI normalized'] = -src_bounding['Upper CI normalized']
src_bounding['sediment yield normalized (kt / km^2 / yr)'] = src_bounding['sediment yield normalized (t / km^2 / yr)']/1000

base = alt.Chart(src_bounding).encode(alt.X("Valley Name", axis=alt.Axis(labelAngle = -60), sort=valley_sorting)).properties(width=fig_width, height=fig_height)
volume = base.mark_bar().encode(
    alt.Y("Annual Mass Wasted normalized:Q", title='Specifc sediment yield (m/yr)', sort=valley_sorting, scale=alt.Scale(
        domain=volume_domain, 
        nice=False
    )),
    alt.Color("Fork of the Nooksack River:N")
)
sedyield = base.mark_bar().encode(
    alt.Y("sediment yield normalized (kt / km^2 / yr):Q", sort=valley_sorting, scale=alt.Scale(
            domain=yield_domain, 
            nice=False
        ), 
        title='Specific sediment yield (kt / km² / yr)'
    ),
    alt.Color("Fork of the Nooksack River:N", scale = alt.Scale(
        domain = [
            'Does not drain to Nooksack River',
            "North Fork",
            "Middle Fork"
        ],
        range = ['#282828', '#808080' , '#DCDCDC'])
    )
)
error_bars = base.mark_bar(width=2, color='black', stroke='white').encode(
        alt.Y("Lower CI normalized:Q", scale=alt.Scale(
                domain=volume_domain, 
                nice=False
            ), 
            title='', axis=alt.Axis(labels=False)
        ),
    alt.Y2("Upper CI normalized:Q", title='')
)
alt.layer(
    volume, 
    sedyield, 
    error_bars
).resolve_scale(y='independent').configure_legend(
    titleFontSize=12,
    labelFontSize=12,
    orient='top'
).configure_axis(
    labelFontSize=12,
    titleFontSize=14,
    titleFontWeight='normal'
)

# %%


fig_width=200
fig_height=300
yield_domain = [-50, 168.99999999999997]
volume_domain = [-29585.79881656805/1000, 100]

src = net_measurements.drop(columns=['time interval']).drop(columns='index').drop(columns=[0])
src['Annual Mass Wasted'] = -src['Annual Mass Wasted']/1000
src['Lower CI'] = -src['Lower CI']/1000
src['Upper CI'] = -src['Upper CI']/1000
src['Lower CI sediment yield'] = -src['Lower CI sediment yield']/1000
src['Upper CI sediment yield'] = -src['Upper CI sediment yield']/1000

src['sediment yield (t / yr)'] = src['sediment yield (t / yr)']/1000

src_bounding = src[src['named interval'] == 'bounding']
base = alt.Chart(src_bounding).encode(alt.X("Valley Name", axis=alt.Axis(labelAngle = -60), sort=valley_sorting)).properties(width=fig_width, height=fig_height)
volume = base.mark_bar().encode(
    alt.Y("Annual Mass Wasted:Q", title='Annual sediment yield (10³ m³/yr)', sort=valley_sorting, scale=alt.Scale(domain=volume_domain, nice=False)),
    alt.Color("Fork of the Nooksack River:N")
)
sedyield = base.mark_bar().encode(
    alt.Y("sediment yield (t / yr):Q", sort=valley_sorting, scale=alt.Scale(domain=yield_domain, nice=False), title='Annual sediment yield (kt / yr)'),
    alt.Color("Fork of the Nooksack River:N", scale = alt.Scale(
        domain = [
            'Does not drain to Nooksack River',
            "North Fork",
            "Middle Fork"
        ],
        range = ['#282828', '#808080' , '#DCDCDC'])
    )
)
error_bars = base.mark_bar(width=2, color='black', stroke='white').encode(
        alt.Y("Lower CI:Q", scale=alt.Scale(domain=volume_domain, nice=False), title='', axis=alt.Axis(labels=False)),
    alt.Y2("Upper CI:Q", title='')
)
bounding_chart = alt.layer(volume, sedyield, error_bars).resolve_scale(y='independent')

src_advance = src[src['named interval'] == 'advance']
base = alt.Chart(src_advance).encode(alt.X("Valley Name", axis=alt.Axis(labelAngle = -60), sort=valley_sorting)).properties(width=fig_width, height=fig_height)
volume = base.mark_bar().encode(
    alt.Y("Annual Mass Wasted:Q", title='Annual sediment yield (10³ m³/yr)', sort=valley_sorting, scale=alt.Scale(domain=volume_domain, nice=False)),
    alt.Color("Fork of the Nooksack River:N")
)
sedyield = base.mark_bar().encode(
    alt.Y("sediment yield (t / yr):Q", sort=valley_sorting, scale=alt.Scale(domain=yield_domain, nice=False), title='Annual sediment yield (kt / yr)'),
    alt.Color("Fork of the Nooksack River:N")
)
error_bars = base.mark_bar(width=2, color='black').encode(
        alt.Y("Lower CI:Q", scale=alt.Scale(domain=volume_domain, nice=False), title='', axis=alt.Axis(labels=False)),
    alt.Y2("Upper CI:Q", title='')
)
advance_chart = alt.layer(volume, sedyield, error_bars).resolve_scale(y='independent')


src_retreat = src[src['named interval'] == 'retreat']
base = alt.Chart(src_retreat).encode(alt.X("Valley Name", axis=alt.Axis(labelAngle = -60), sort=valley_sorting)).properties(width=fig_width, height=fig_height)
volume = base.mark_bar().encode(
    alt.Y("Annual Mass Wasted:Q", title='Annual sediment yield (10³ m³/yr)', sort=valley_sorting, scale=alt.Scale(domain=volume_domain, nice=False)),
    alt.Color("Fork of the Nooksack River:N")
)
sedyield = base.mark_bar().encode(
    alt.Y("sediment yield (t / yr):Q", sort=valley_sorting, scale=alt.Scale(domain=yield_domain, nice=False), title='Annual sediment yield (kt / yr)'),
    alt.Color("Fork of the Nooksack River:N")
)
error_bars = base.mark_bar(width=2, color='black').encode(
    alt.Y("Lower CI:Q", scale=alt.Scale(domain=volume_domain, nice=False), title='', axis=alt.Axis(labels=False)),
    alt.Y2("Upper CI:Q", title='')
)
retreat_chart = alt.layer(volume, sedyield, error_bars).resolve_scale(y='independent')

(
    bounding_chart | 
    advance_chart | 
    retreat_chart
).configure_legend(titleFontSize=12, labelFontSize=12, orient='top').configure_axis(labelFontSize=12, titleFontSize=14, titleFontWeight='normal')

# %% [markdown]
# ## Save data to csv

# %%
src[[
'Valley Name',
'Fork of the Nooksack River',
'named interval',
'Annual Mass Wasted',
'Lower CI',
'Upper CI',
'Lower CI sediment yield',
'Upper CI sediment yield',
'sediment yield (t / yr)',
]].to_csv('outputs/final_figures_data/volumes_and_yields_per_valley.csv')

# %% [markdown]
# ### Merge in longitudinal slope data

# %%
net_measurements = net_measurements.merge(
    long_slopes_df, 
    on='Valley Name'
)

# %%
net_measurements

# %% [markdown]
# ## Keep only the bounding data

# %%
net_measurements = net_measurements[net_measurements['named interval'] == 'bounding']

# %%
net_measurements = net_measurements[[
    'Valley Name',
    'watershed area (square km)', # A, could also use the column "drainage area (km)" instead
    'longitudinal slope 2015', # S_c
    'slope hillslope erosion area', # S_h
    'glacial retreat absolute', # ∆A_g
    'glacial area 1977/79', #A_g
    'nonigneous fraction',
    'sediment yield (t / yr)', #Q_s
    'sediment yield normalized (t / km^2 / yr)',
    'Upper CI sediment yield',
    'Lower CI sediment yield',
    'Upper CI sediment yield normalized',
    'Lower CI sediment yield normalized'
]]
net_measurements['glacial retreat absolute'] = -net_measurements['glacial retreat absolute']
net_measurements['glacial retreat relative'] = net_measurements['glacial retreat absolute'] / net_measurements['glacial area 1977/79']

# %%
net_measurements

# %% [markdown]
# # Plot Scatterplots

# %% [markdown]
# ## Prep Data

# %% [markdown]
# ### rename columns (for plotting convenience)

# %%
net_measurements = net_measurements.rename(columns={
    'sediment yield (t / yr)': 'Sediment Yield (ton/yr)',
    'watershed area (square km)': 'Drainage area (square km)',
    'longitudinal slope 2015': 'Channel slope',
    'slope hillslope erosion area': 'Hillslope domain slope',
    'glacial retreat absolute': 'Glacial retreat area (km²)',
    'nonigneous fraction': 'Nonigneous fraction',  
    'sediment yield normalized (t / km^2 / yr)' : 'Sediment Yield (ton/km²/yr)' 
})


net_measurements

# %% [markdown]
# ## Plot sediment yield vs explanatory variables

# %%
circles_y = alt.Chart().mark_point(size=100, strokeWidth=2).encode(
    alt.Y('Sediment Yield (ton/yr):Q'),
    alt.Color('Valley Name:N')
)
points_y = alt.Chart().mark_circle(size=110).encode(
    alt.Y('Sediment Yield (ton/yr):Q'),
    alt.Color('Valley Name:N')
)
bars_y = alt.Chart().mark_line().encode(
    alt.Y('Lower CI sediment yield'),
    alt.Y2('Upper CI sediment yield'),
    alt.Color('Valley Name:N')
)

darea = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Drainage area (square km):Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Drainage area (square km):Q')), 
    bars_y.encode(alt.X('Drainage area (square km):Q')), 
    data=net_measurements
).properties(
    width=200, height=200
)

channelslope = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Channel slope:Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Channel slope:Q')), 
    bars_y.encode(alt.X('Channel slope:Q', scale=alt.Scale(zero=False))),
    data=net_measurements
).properties(
    width=200, height=200
)

hillslope = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Hillslope domain slope:Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Hillslope domain slope:Q')), 
    bars_y.encode(alt.X('Hillslope domain slope:Q', scale=alt.Scale(zero=False))),
    data=net_measurements
).properties(
    width=200, height=200
)

glacialretreat = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Glacial retreat area (km²):Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Glacial retreat area (km²):Q')), 
    bars_y.encode(alt.X('Glacial retreat area (km²):Q')),
    data=net_measurements
).properties(
    width=200, height=200
)

lithology = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Nonigneous fraction:Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/yr)', 0)).encode(alt.X('Nonigneous fraction:Q')), 
    bars_y.encode(alt.X('Nonigneous fraction:Q')),
    data=net_measurements
).properties(
    width=200, height=200
)

darea | channelslope | hillslope | glacialretreat | lithology

# %% [markdown]
# ## Plot specific sediment yield vs explanatory variables

# %%
circles_y = alt.Chart().mark_point(size=100, strokeWidth=2).encode(
    alt.Y('Sediment Yield (ton/km²/yr):Q'),
    alt.Color('Valley Name:N')
)
points_y = alt.Chart().mark_circle(size=110).encode(
    alt.Y('Sediment Yield (ton/km²/yr):Q'),
    alt.Color('Valley Name:N')
)
bars_y = alt.Chart().mark_line().encode(
    alt.Y('Lower CI sediment yield normalized'),
    alt.Y2('Upper CI sediment yield normalized'),
    alt.Color('Valley Name:N')
)

darea_normalized = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Drainage area (square km):Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Drainage area (square km):Q')), 
    bars_y.encode(alt.X('Drainage area (square km):Q')), 
    data=net_measurements
).properties(
    width=200, height=200
)

channelslope_normalized = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Channel slope:Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Channel slope:Q')), 
    bars_y.encode(alt.X('Channel slope:Q', scale=alt.Scale(zero=False))),
    data=net_measurements
).properties(
    width=200, height=200
)

hillslope_normalized = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Hillslope domain slope:Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Hillslope domain slope:Q')), 
    bars_y.encode(alt.X('Hillslope domain slope:Q', scale=alt.Scale(zero=False))),
    data=net_measurements
).properties(
    width=200, height=200
)

glacialretreat_normalized = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Glacial retreat area (km²):Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Glacial retreat area (km²):Q')), 
    bars_y.encode(alt.X('Glacial retreat area (km²):Q')),
    data=net_measurements
).properties(
    width=200, height=200
)

lithology_normalized = alt.layer(
    points_y.transform_filter(alt.FieldGTEPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Nonigneous fraction:Q')), 
    circles_y.transform_filter(alt.FieldLTPredicate('Sediment Yield (ton/km²/yr)', 0)).encode(alt.X('Nonigneous fraction:Q')), 
    bars_y.encode(alt.X('Nonigneous fraction:Q')),
    data=net_measurements
).properties(
    width=200, height=200
)

darea_normalized | channelslope_normalized | hillslope_normalized | glacialretreat_normalized | lithology_normalized

# %% [markdown]
# ## Plot explanatory variables interactions

# %%
(
    alt.Chart(net_measurements).mark_circle(size=100).encode(
        alt.Y('Drainage area (square km):Q', title='Drainage area (km²)', scale=alt.Scale(zero=False)),
        alt.X("Nonigneous fraction:Q",),
        alt.Color("Valley Name:N")
    ).properties(
        width=200, height=200
    ) | \
    alt.Chart(net_measurements).mark_circle(size=100).encode(
        alt.Y('Channel slope:Q', scale=alt.Scale(zero=False)),
        alt.X("Nonigneous fraction:Q",),
        alt.Color("Valley Name:N")
    ).properties(
        width=200, height=200
    ) | \
    alt.Chart(net_measurements).mark_circle(size=100).encode(
        alt.Y('Hillslope domain slope:Q', scale=alt.Scale(zero=False)),
        alt.X("Nonigneous fraction:Q",),
        alt.Color("Valley Name:N")
    ).properties(
        width=200, height=200
    ) | \
    alt.Chart(net_measurements).mark_circle(size=100).encode(
        alt.Y('Glacial retreat area (km²):Q', scale=alt.Scale(zero=False)),
        alt.X("Nonigneous fraction:Q",),
        alt.Color("Valley Name:N")
    ).properties(
        width=200, height=200
    )


).configure_legend(
    titleFontSize=22, labelFontSize=22, orient='top'
).configure_axis(
    labelFontSize=16, titleFontSize=16
).configure_title(
    fontSize=22
)

# %% [markdown]
# # Modeling

# %%
model_data = net_measurements.copy()

# %% [markdown]
# ## Save Model Data to CSV

# %%
model_data.to_csv("outputs/modeling_powerlaw_data.csv")

# %% [markdown]
# ## Remove net depositional data points

# %%
model_data = model_data[model_data["Sediment Yield (ton/yr)"] > 0]

# %%
model_data['Valley Name'].unique()

# %% [markdown]
# ## Remove two outlier basins

# %%
# model_data = model_data[~model_data['Valley Name'].isin(['Boulder', 'Thunder'])]

# %%
model_data['Valley Name'].unique()

# %% [markdown]
# ## Create Models

# %%
parameters_dict_powerlaw = {}
parameters_dict_linear = {}

parameters_dict_powerlaw_normalized = {}
parameters_dict_linear_normalized = {}


# %% [markdown]
# ### Define Power law models

# %%
def model_1_powerlaw(x, k, l):
    """
    Args:
        x (float): drainage area
    """
    return k*x**l

def model_2_powerlaw(x, k, m):
    """
    Args:
        x (float): channel slope
    """
    return k*x**m

def model_3_powerlaw(x, k, n):
    """
    Args:
        x (float): hillslope slope
    """
    return k*x**n

def model_4_powerlaw(x, k, p):
    """
    Args:
        x (float): glacier area change
    """
    return k*x**p

def model_5_powerlaw(x, k, q):
    """
    Args:
        x (float): nonigneous fraction
    """
    return k*x**q

def model_6_powerlaw(x, k, l, n):
    """
    Args:
        x (list(float))): [drainage area, hillslope slope]
    """
    return k*(x[0]**l)*(x[1]**n)

def model_7_powerlaw(x, k, l, n, p):
    """
    Args:
        x (list(float))): [drainage area, hillslope slope, glacier area change]
    """
    return k*(x[0]**l)*(x[1]**n)*(x[2]**p)

# def model7(x, k, l, m, n):
#     """
#     Args:
#         x (list(float))): [drainage area, channel slope, hillslope slope]
#     """
#     return k*(x[0]**l)*(x[1]**m)*(x[2]**n)

# def model8(x, k, l, m, n, p):
#     """
#     Args:
#         x (list(float))): [drainage area, channel slope, hillslope slope, glacier area change]
#     """
#     return k*(x[0]**l)*(x[1]**m)*(x[2]**n)*(x[3]**p)


# %% [markdown]
# ### Define linear models

# %%
def model_1_linear(x, l, i):
    """
    Args:
        x (float): drainage area
    """
    return l*x + i

def model_2_linear(x, m, i):
    """
    Args:
        x (float): channel slope
    """
    return m*x + i

def model_3_linear(x, n, i):
    """
    Args:
        x (float): hillslope slope
    """
    return n*x

def model_4_linear(x, p, i):
    """
    Args:
        x (float): glacier area change
    """
    return p*x + i

def model_5_linear(x, q, i):
    """
    Args:
        x (float): nonigneous fraction
    """
    return q*x + i

def model_6_linear(x, l, n, i):
    """
    Args:
        x (list(float))): [drainage area, hillslope slope]
    """
    return l*x[0] + n*x[1] + i

def model_7_linear(x, l, n, p, i):
    """
    Args:
        x (list(float))): [drainage area, hillslope slope, glacier area change]
    """
    return l*x[0] + n*x[1] + p*x[2] + i


# %% [markdown]
# ## Run Models

# %% [markdown]
# #### Model 1

# %%
model_1_data = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_1_powerlaw, 
    model_1_data['Drainage area (square km)'].to_list(),
    model_1_data['Sediment Yield (ton/yr)'].to_list()
)

popt_powerlaw = [385.93, 2.0287]

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_1_linear, 
    model_1_data['Drainage area (square km)'].to_list(),
    model_1_data['Sediment Yield (ton/yr)'].to_list()
)

model_1_data['Power law predicted sediment yield (ton/yr)'] = model_1_data.apply(
    lambda row: model_1_powerlaw(row['Drainage area (square km)'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_1_data['Linear predicted sediment yield (ton/yr)'] = model_1_data.apply(
    lambda row: model_1_linear(row['Drainage area (square km)'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw[1] = popt_powerlaw
parameters_dict_linear[1] = popt_linear

model_1_plot_linear = r2_plot(model_1_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_1_plot_powerlaw = r2_plot(model_1_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_1_plot_linear | model_1_plot_powerlaw

# %% [markdown]
# #### Model 1 SSY

# %%
model_1_data_normalized = model_data.copy()

popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_1_powerlaw, 
    model_1_data_normalized['Drainage area (square km)'].to_list(),
    model_1_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_1_linear, 
    model_1_data_normalized['Drainage area (square km)'].to_list(),
    model_1_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_1_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_1_data_normalized.apply(
    lambda row: model_1_powerlaw(row['Drainage area (square km)'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_1_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_1_data_normalized.apply(
    lambda row: model_1_linear(row['Drainage area (square km)'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw_normalized[1] = popt_powerlaw
parameters_dict_linear_normalized[1] = popt_linear

model_1_plot_linear_normalized = r2_plot(model_1_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_1_plot_powerlaw_normalized = r2_plot(model_1_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)

model_1_plot_linear_normalized | model_1_plot_powerlaw_normalized

# %% [markdown]
# #### Model 2

# %%
model_2_data = model_data.copy()

popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_2_powerlaw, 
    model_2_data['Channel slope'].to_list(),
    model_2_data['Sediment Yield (ton/yr)'].to_list(),
    [106, -2.9],
    method = 'trf'
)

popt_powerlaw = [106.36, -2.9]

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_2_linear, 
    model_2_data['Channel slope'].to_list(),
    model_2_data['Sediment Yield (ton/yr)'].to_list()
)

model_2_data['Power law predicted sediment yield (ton/yr)'] = model_2_data.apply(
    lambda row: model_2_powerlaw(row['Channel slope'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_2_data['Linear predicted sediment yield (ton/yr)'] = model_2_data.apply(
    lambda row: model_2_linear(row['Channel slope'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw[2] = popt_powerlaw
parameters_dict_linear[2] = popt_linear

model_2_plot_linear = r2_plot(model_2_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_2_plot_powerlaw = r2_plot(model_2_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_2_plot_linear | model_2_plot_powerlaw

# %% [markdown]
# #### Model 2 SSY

# %%
model_2_data_normalized = model_data.copy()

popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_2_powerlaw, 
    model_2_data_normalized['Channel slope'].to_list(),
    model_2_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_2_linear, 
    model_2_data_normalized['Channel slope'].to_list(),
    model_2_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_2_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_2_data_normalized.apply(
    lambda row: model_2_powerlaw(row['Channel slope'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_2_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_2_data_normalized.apply(
    lambda row: model_2_linear(row['Channel slope'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw_normalized[2] = popt_powerlaw
parameters_dict_linear_normalized[2] = popt_linear

model_2_plot_linear_normalized = r2_plot(model_2_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_2_plot_powerlaw_normalized = r2_plot(model_2_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)

model_2_plot_linear_normalized | model_2_plot_powerlaw_normalized

# %% [markdown]
# #### Model 3

# %%
model_3_data = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_3_powerlaw, 
    model_3_data['Hillslope domain slope'].to_list(),
    model_3_data['Sediment Yield (ton/yr)'].to_list()
)

popt_powerlaw = [318055, 3.9142]

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_3_linear, 
    model_3_data['Hillslope domain slope'].to_list(),
    model_3_data['Sediment Yield (ton/yr)'].to_list()
)

model_3_data['Power law predicted sediment yield (ton/yr)'] = model_3_data.apply(
    lambda row: model_3_powerlaw(row['Hillslope domain slope'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_3_data['Linear predicted sediment yield (ton/yr)'] = model_3_data.apply(
    lambda row: model_3_linear(row['Hillslope domain slope'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw[3] = popt_powerlaw
parameters_dict_linear[3] = popt_linear

model_3_plot_linear = r2_plot(model_3_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_3_plot_powerlaw = r2_plot(model_3_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_3_plot_linear | model_3_plot_powerlaw

# %% [markdown]
# #### Model 3 SSY

# %%
model_3_data_normalized = model_data.copy()

popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_3_powerlaw, 
    model_3_data_normalized['Hillslope domain slope'].to_list(),
    model_3_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_3_linear, 
    model_3_data_normalized['Hillslope domain slope'].to_list(),
    model_3_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_3_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_3_data_normalized.apply(
    lambda row: model_3_powerlaw(row['Hillslope domain slope'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_3_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_3_data_normalized.apply(
    lambda row: model_3_linear(row['Hillslope domain slope'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw_normalized[3] = popt_powerlaw
parameters_dict_linear_normalized[3] = popt_linear

model_3_plot_linear_normalized = r2_plot(model_3_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_3_plot_powerlaw_normalized = r2_plot(model_3_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)

model_3_plot_linear_normalized | model_3_plot_powerlaw_normalized

# %% [markdown]
# #### Model 4

# %%
model_4_data = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_4_powerlaw, 
    model_4_data['Glacial retreat area (km²)'].to_list(),
    model_4_data['Sediment Yield (ton/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_4_linear, 
    model_4_data['Glacial retreat area (km²)'].to_list(),
    model_4_data['Sediment Yield (ton/yr)'].to_list()
)

model_4_data['Power law predicted sediment yield (ton/yr)'] = model_4_data.apply(
    lambda row: model_4_powerlaw(row['Glacial retreat area (km²)'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_4_data['Linear predicted sediment yield (ton/yr)'] = model_4_data.apply(
    lambda row: model_4_linear(row['Glacial retreat area (km²)'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw[4] = popt_powerlaw
parameters_dict_linear[4] = popt_linear

model_4_plot_linear = r2_plot(model_4_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_4_plot_powerlaw = r2_plot(model_4_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_4_plot_linear | model_4_plot_powerlaw

# %% [markdown]
# #### Model 4 SSY

# %%
model_4_data_normalized = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_4_powerlaw, 
    model_4_data_normalized['Glacial retreat area (km²)'].to_list(),
    model_4_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_4_linear, 
    model_4_data_normalized['Glacial retreat area (km²)'].to_list(),
    model_4_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_4_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_4_data_normalized.apply(
    lambda row: model_4_powerlaw(row['Glacial retreat area (km²)'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_4_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_4_data_normalized.apply(
    lambda row: model_4_linear(row['Glacial retreat area (km²)'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw_normalized[4] = popt_powerlaw
parameters_dict_linear_normalized[4] = popt_linear

model_4_plot_linear_normalized = r2_plot(model_4_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_4_plot_powerlaw_normalized = r2_plot(model_4_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)

model_4_plot_linear_normalized | model_4_plot_powerlaw_normalized

# %% [markdown]
# #### Model 5

# %%
model_5_data = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_5_powerlaw, 
    model_5_data['Nonigneous fraction'].to_list(),
    model_5_data['Sediment Yield (ton/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_5_linear, 
    model_5_data['Nonigneous fraction'].to_list(),
    model_5_data['Sediment Yield (ton/yr)'].to_list()
)

model_5_data['Power law predicted sediment yield (ton/yr)'] = model_5_data.apply(
    lambda row: model_5_powerlaw(row['Nonigneous fraction'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_5_data['Linear predicted sediment yield (ton/yr)'] = model_5_data.apply(
    lambda row: model_5_linear(row['Nonigneous fraction'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw[5] = popt_powerlaw
parameters_dict_linear[5] = popt_linear

model_5_plot_linear = r2_plot(model_5_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_5_plot_powerlaw = r2_plot(model_5_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_5_plot_linear | model_5_plot_powerlaw

# %% [markdown]
# #### Model 5 SSY

# %%
model_5_data_normalized = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_5_powerlaw, 
    model_5_data_normalized['Nonigneous fraction'].to_list(),
    model_5_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_5_linear, 
    model_5_data_normalized['Nonigneous fraction'].to_list(),
    model_5_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_5_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_5_data_normalized.apply(
    lambda row: model_5_powerlaw(row['Nonigneous fraction'], popt_powerlaw[0], popt_powerlaw[1]),
    axis=1
)

model_5_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_5_data_normalized.apply(
    lambda row: model_5_linear(row['Nonigneous fraction'], popt_linear[0], popt_linear[1]),
    axis=1
)

parameters_dict_powerlaw_normalized[5] = popt_powerlaw
parameters_dict_linear_normalized[5] = popt_linear

model_5_plot_linear_normalized = r2_plot(model_5_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_5_plot_powerlaw_normalized = r2_plot(model_5_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)

model_5_plot_linear_normalized | model_5_plot_powerlaw_normalized

# %% [markdown]
# #### Model 6

# %%
model_6_data = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_6_powerlaw, 
    np.array([
        model_6_data['Drainage area (square km)'].to_list(),
        model_6_data['Hillslope domain slope'].to_list(),
    ]),
    model_6_data['Sediment Yield (ton/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_6_linear, 
    np.array([
        model_6_data['Drainage area (square km)'].to_list(),
        model_6_data['Hillslope domain slope'].to_list(),
    ]),
    model_6_data['Sediment Yield (ton/yr)'].to_list()
)

model_6_data['Power law predicted sediment yield (ton/yr)'] = model_6_data.apply(
    lambda row: model_6_powerlaw(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope']
        ), popt_powerlaw[0], popt_powerlaw[1], popt_powerlaw[2]
    ),
    axis=1
)

model_6_data['Linear predicted sediment yield (ton/yr)'] = model_6_data.apply(
    lambda row: model_6_linear(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope']
        ), popt_linear[0], popt_linear[1], popt_linear[2]
    ),
    axis=1
)

parameters_dict_powerlaw[6] = popt_powerlaw
parameters_dict_linear[6] = popt_linear

model_6_plot_linear = r2_plot(model_6_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_6_plot_powerlaw = r2_plot(model_6_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_6_plot_powerlaw_error_bars = alt.Chart(model_6_data).mark_rule(color='#1f77b4').encode(
    alt.X('Power law predicted sediment yield (ton/yr)'),
    alt.Y('Lower CI sediment yield'),
    alt.Y2('Upper CI sediment yield')
)

model_6_plot_linear | model_6_plot_powerlaw+model_6_plot_powerlaw_error_bars

# %% [markdown]
# #### Model 6 SSY

# %%
model_6_data_normalized = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_6_powerlaw, 
    np.array([
        model_6_data_normalized['Drainage area (square km)'].to_list(),
        model_6_data_normalized['Hillslope domain slope'].to_list(),
    ]),
    model_6_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_6_linear, 
    np.array([
        model_6_data_normalized['Drainage area (square km)'].to_list(),
        model_6_data_normalized['Hillslope domain slope'].to_list(),
    ]),
    model_6_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_6_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_6_data_normalized.apply(
    lambda row: model_6_powerlaw(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope']
        ), popt_powerlaw[0], popt_powerlaw[1], popt_powerlaw[2]
    ),
    axis=1
)

model_6_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_6_data_normalized.apply(
    lambda row: model_6_linear(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope']
        ), popt_linear[0], popt_linear[1], popt_linear[2]
    ),
    axis=1
)

parameters_dict_powerlaw_normalized[6] = popt_powerlaw
parameters_dict_linear_normalized[6] = popt_linear

model_6_plot_linear_normalized = r2_plot(model_6_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_6_plot_powerlaw_normalized = r2_plot(model_6_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)



model_6_plot_powerlaw_error_bars_normalized = alt.Chart(model_6_data_normalized).mark_rule(color='#1f77b4').encode(
    alt.X('Power law predicted sediment yield (ton/km²/yr)', title=''),
    alt.Y('Lower CI sediment yield normalized'),
    alt.Y2('Upper CI sediment yield normalized')
)
model_6_plot_linear_normalized | model_6_plot_powerlaw_normalized+model_6_plot_powerlaw_error_bars_normalized

# %% [markdown]
# #### Model 7
#

# %%
model_7_data = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_7_powerlaw, 
    np.array([
        model_7_data['Drainage area (square km)'].to_list(),
        model_7_data['Hillslope domain slope'].to_list(),
        model_7_data['Glacial retreat area (km²)'].to_list(),
    ]),
    model_7_data['Sediment Yield (ton/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_7_linear, 
    np.array([
        model_7_data['Drainage area (square km)'].to_list(),
        model_7_data['Hillslope domain slope'].to_list(),
        model_7_data['Glacial retreat area (km²)'].to_list(),
    ]),
    model_7_data['Sediment Yield (ton/yr)'].to_list()
)

model_7_data['Power law predicted sediment yield (ton/yr)'] = model_7_data.apply(
    lambda row: model_7_powerlaw(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope'],
            row['Glacial retreat area (km²)'],
        ), popt_powerlaw[0], popt_powerlaw[1], popt_powerlaw[2], popt_powerlaw[3]
    ),
    axis=1
)

model_7_data['Linear predicted sediment yield (ton/yr)'] = model_7_data.apply(
    lambda row: model_7_linear(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope'],
            row['Glacial retreat area (km²)'],
        ), popt_linear[0], popt_linear[1], popt_linear[2], popt_linear[3]
    ),
    axis=1
)

parameters_dict_powerlaw[7] = popt_powerlaw
parameters_dict_linear[7] = popt_linear

model_7_plot_linear = r2_plot(model_7_data, 'Linear predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)
model_7_plot_powerlaw = r2_plot(model_7_data, 'Power law predicted sediment yield (ton/yr)', 'Sediment Yield (ton/yr)', limit=100000)

model_7_plot_linear | model_7_plot_powerlaw

# %% [markdown]
# #### Model 7 SSY

# %%
model_7_data_normalized = model_data.copy()



popt_powerlaw, pcov_powerlaw = scipy.optimize.curve_fit(
    model_7_powerlaw, 
    np.array([
        model_7_data_normalized['Drainage area (square km)'].to_list(),
        model_7_data_normalized['Hillslope domain slope'].to_list(),
        model_7_data_normalized['Glacial retreat area (km²)'].to_list(),
    ]),
    model_7_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

popt_linear, pcov_linear = scipy.optimize.curve_fit(
    model_7_linear, 
    np.array([
        model_7_data_normalized['Drainage area (square km)'].to_list(),
        model_7_data_normalized['Hillslope domain slope'].to_list(),
        model_7_data_normalized['Glacial retreat area (km²)'].to_list(),
    ]),
    model_7_data_normalized['Sediment Yield (ton/km²/yr)'].to_list()
)

model_7_data_normalized['Power law predicted sediment yield (ton/km²/yr)'] = model_7_data_normalized.apply(
    lambda row: model_7_powerlaw(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope'],
            row['Glacial retreat area (km²)'],
        ), popt_powerlaw[0], popt_powerlaw[1], popt_powerlaw[2], popt_powerlaw[3]
    ),
    axis=1
)

model_7_data_normalized['Linear predicted sediment yield (ton/km²/yr)'] = model_7_data_normalized.apply(
    lambda row: model_7_linear(
        (
            row['Drainage area (square km)'],
            row['Hillslope domain slope'],
            row['Glacial retreat area (km²)'],
        ), popt_linear[0], popt_linear[1], popt_linear[2], popt_linear[3]
    ),
    axis=1
)

parameters_dict_powerlaw_normalized[7] = popt_powerlaw
parameters_dict_linear_normalized[7] = popt_linear

model_7_plot_linear_normalized = r2_plot(model_7_data_normalized, 'Linear predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)
model_7_plot_powerlaw_normalized = r2_plot(model_7_data_normalized, 'Power law predicted sediment yield (ton/km²/yr)', 'Sediment Yield (ton/km²/yr)', limit=8000)

model_7_plot_linear_normalized | model_7_plot_powerlaw_normalized


# %% [markdown]
# # Plot Model Results

# %%
def add_props(plot, params, data, model_n, predicted_col, nse_label = "NSE: "):
    return plot.properties(
        title={
            'subtitle': [
                str([np.round(var, 2) for var in params]),
                nse_label + str(np.round(nse(data['Sediment Yield (ton/yr)'], data[predicted_col]), 3))],
            'text': f"Model {str(model_n)}"
        }
    )


# %% [markdown]
# ## Multivariate models - Observed vs Predicted Plots

# %%
model_6_plot_powerlaw_normalized

# %%
model_6_ssy_results = add_props(
    model_6_plot_powerlaw_normalized + model_6_plot_powerlaw_error_bars_normalized, parameters_dict_powerlaw_normalized[6], model_6_data_normalized, 6, predicted_col='Power law predicted sediment yield (ton/km²/yr)', nse_label='NSE (Power law Model): '
).encode(
    alt.X(title="Predicted sediment yield (ton/km²/yr)"),
    alt.Y(title='Observed sediment yield (ton/km²/yr)')
)

model_6_results = add_props(
    model_6_plot_powerlaw + model_6_plot_powerlaw_error_bars, parameters_dict_powerlaw[6], model_6_data, 6, predicted_col='Power law predicted sediment yield (ton/yr)', nse_label='NSE (Power law Model): '
).encode(
    alt.X(title="Predicted sediment yield (ton/yr)"),
    alt.Y(title='Observed sediment yield (ton/yr)')
)

(model_6_ssy_results | model_6_results).configure_legend(
    titleFontSize=22, labelFontSize=22, orient='right'
).configure_axis(
    labelFontSize=16, titleFontSize=16
).configure_title(
    fontSize=22
)


# %% [markdown]
# ## Single variate models - Observed vs Variabel Plots

# %% [markdown]
# ### Sediment Yield

# %%
def add_props_2models(plot, data, model_n, params_linear, params_powerlaw):
    return plot.properties(
        width=200, height=200,
        title={
            'subtitle': [
                "Parameters (Linear model): " + str([np.round(var, 2) for var in params_linear]),
                "Parameters (Power law model): " + str([np.round(var, 2) for var in params_powerlaw]),
                "NSE (Linear model): " + str(np.round(nse(data['Sediment Yield (ton/yr)'], data['Linear predicted sediment yield (ton/yr)']), 2)),
                # "r² (Linear model): " + str(np.round(r2_score(data['Sediment Yield (ton/yr)'], data['Linear predicted sediment yield (ton/yr)']), 2)),
                "NSE (Power law model): " + str(np.round(nse(data['Sediment Yield (ton/yr)'], data['Power law predicted sediment yield (ton/yr)']), 2))
            ],
            'text': f"Model {str(model_n)}"
        }
    )


# %% [markdown]
# #### Model 1

# %%

domain_space = pd.Series(np.linspace(0,14,100))
model_1_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_1_linear(x, parameters_dict_linear[1][0], parameters_dict_linear[1][1])),
    'Power law model': domain_space.apply(lambda x: model_1_powerlaw(x, parameters_dict_powerlaw[1][0], parameters_dict_powerlaw[1][1])) 
})

linear_model = alt.Chart(model_1_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Drainage area (km²)'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-40000, 120000)))
)

powerlaw_model = alt.Chart(model_1_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_1_scatterplot = (linear_model + powerlaw_model + darea)

model_1_scatterplot = add_props_2models(model_1_scatterplot, model_1_data, 1, parameters_dict_linear[1], parameters_dict_powerlaw[1])
model_1_scatterplot

# %% [markdown]
# #### Model 2

# %%

domain_space = pd.Series(np.linspace(0.05,0.30,100))
model_2_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_2_linear(x, parameters_dict_linear[2][0], parameters_dict_linear[2][1])),
    'Power law model': domain_space.apply(lambda x: model_2_powerlaw(x, parameters_dict_powerlaw[2][0], parameters_dict_powerlaw[2][1])) 
})

linear_model = alt.Chart(model_2_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Channel slope'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-40000, 120000), clamp=True))
)

powerlaw_model = alt.Chart(model_2_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_2_scatterplot = (linear_model + powerlaw_model + channelslope)

model_2_scatterplot = add_props_2models(model_2_scatterplot, model_2_data, 2, parameters_dict_linear[2], parameters_dict_powerlaw[2])

# %% [markdown]
# #### Model 3

# %%

domain_space = pd.Series(np.linspace(0.4,0.9,100))
model_3_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_3_linear(x, parameters_dict_linear[3][0], parameters_dict_linear[3][1])),
    'Power law model': domain_space.apply(lambda x: model_3_powerlaw(x, parameters_dict_powerlaw[3][0], parameters_dict_powerlaw[3][1])) 
})

linear_model = alt.Chart(model_3_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Hillslope domain slope'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-40000, 120000)))
)

powerlaw_model = alt.Chart(model_3_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_3_scatterplot = (linear_model + powerlaw_model + hillslope)

model_3_scatterplot = add_props_2models(model_3_scatterplot, model_3_data, 3, parameters_dict_linear[3], parameters_dict_powerlaw[3])

# %% [markdown]
# #### Model 4

# %%

domain_space = pd.Series(np.linspace(0.0,0.9,100))
model_4_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_4_linear(x, parameters_dict_linear[4][0], parameters_dict_linear[4][1])),
    'Power law model': domain_space.apply(lambda x: model_4_powerlaw(x, parameters_dict_powerlaw[4][0], parameters_dict_powerlaw[4][1])) 
})

linear_model = alt.Chart(model_4_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Glacial retreat area (km²)'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-40000, 120000)))
)

powerlaw_model = alt.Chart(model_4_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

# model_4_scatterplot = (linear_model + powerlaw_model + glacialretreat)
model_4_scatterplot = (linear_model + glacialretreat)

model_4_scatterplot = add_props_2models(model_4_scatterplot, model_4_data, 4, parameters_dict_linear[4], parameters_dict_powerlaw[4])

# %% [markdown]
# #### Model 5 

# %%

domain_space = pd.Series(np.linspace(0.0,1.0,100))
model_5_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_5_linear(x, parameters_dict_linear[5][0], parameters_dict_linear[5][1])),
    'Power law model': domain_space.apply(lambda x: model_5_powerlaw(x, parameters_dict_powerlaw[5][0], parameters_dict_powerlaw[5][1])) 
})

linear_model = alt.Chart(model_5_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Nonigneous fraction'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-40000, 120000)), title='')
)

powerlaw_model = alt.Chart(model_5_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_5_scatterplot = linear_model + powerlaw_model + lithology
model_5_scatterplot = linear_model + lithology

model_5_scatterplot = add_props_2models(model_5_scatterplot, model_5_data, 5, parameters_dict_linear[5], parameters_dict_powerlaw[5])

# %%
(
    model_1_scatterplot | model_2_scatterplot | model_3_scatterplot | model_4_scatterplot | model_5_scatterplot
).configure_legend(
    titleFontSize=22, labelFontSize=22, orient='right'
).configure_axis(
    labelFontSize=22, titleFontSize=22
).configure_title(
    fontSize=22
)


# %% [markdown]
# ### Specific Sediment Yield

# %%
def add_props_2models_normalized(plot, data, model_n, params_linear, params_powerlaw):
    return plot.properties(
        width=200, height=200,
        title={
            'subtitle': [
                "Parameters (Linear model): " + str([np.round(var, 2) for var in params_linear]),
                "Parameters (Power law model): " + str([np.round(var, 2) for var in params_powerlaw]),
                "NSE (Linear model): " + str(np.round(nse(data['Sediment Yield (ton/km²/yr)'], data['Linear predicted sediment yield (ton/km²/yr)']), 2)),
                # "r² (Linear model): " + str(np.round(r2_score(data['Sediment Yield (ton/km²/yr)'], data['Linear predicted sediment yield (ton/km²/yr)']), 2)),
                "NSE (Power law model): " + str(np.round(nse(data['Sediment Yield (ton/km²/yr)'], data['Power law predicted sediment yield (ton/km²/yr)']), 2))
            ],
            'text': f"Model {str(model_n)}"
        }
    )


# %% [markdown]
# #### Model 1

# %%

domain_space = pd.Series(np.linspace(0,14,100))
model_1_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_1_linear(x, parameters_dict_linear_normalized[1][0], parameters_dict_linear_normalized[1][1])),
    'Power law model': domain_space.apply(lambda x: model_1_powerlaw(x, parameters_dict_powerlaw_normalized[1][0], parameters_dict_powerlaw_normalized[1][1])) 
})

linear_model = alt.Chart(model_1_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Drainage area (km²)'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-10000, 10000)))
)

powerlaw_model = alt.Chart(model_1_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_1_scatterplot_normed = (linear_model + darea_normalized)

model_1_scatterplot_normed = add_props_2models_normalized(model_1_scatterplot_normed, model_1_data_normalized, 1, parameters_dict_linear_normalized[1], parameters_dict_powerlaw_normalized[1])

# %% [markdown]
# #### Model 2

# %%

domain_space = pd.Series(np.linspace(0.05,0.30,100))
model_2_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_2_linear(x, parameters_dict_linear_normalized[2][0], parameters_dict_linear_normalized[2][1])),
    'Power law model': domain_space.apply(lambda x: model_2_powerlaw(x, parameters_dict_powerlaw_normalized[2][0], parameters_dict_powerlaw_normalized[2][1])) 
})

linear_model = alt.Chart(model_2_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Channel slope'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-10000, 10000), clamp=True))
)

powerlaw_model = alt.Chart(model_2_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_2_scatterplot_normed = (linear_model + channelslope_normalized)

model_2_scatterplot_normed = add_props_2models_normalized(model_2_scatterplot_normed, model_2_data_normalized, 2, parameters_dict_linear_normalized[2], parameters_dict_powerlaw_normalized[2])

# %% [markdown]
# #### Model 3

# %%

domain_space = pd.Series(np.linspace(0.4,0.9,100))
model_3_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_3_linear(x, parameters_dict_linear_normalized[3][0], parameters_dict_linear_normalized[3][1])),
    'Power law model': domain_space.apply(lambda x: model_3_powerlaw(x, parameters_dict_powerlaw_normalized[3][0], parameters_dict_powerlaw_normalized[3][1])) 
})

linear_model = alt.Chart(model_3_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Hillslope domain slope'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-10000, 10000)))
)

powerlaw_model = alt.Chart(model_3_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_3_scatterplot_normed = (linear_model + hillslope_normalized)

model_3_scatterplot_normed = add_props_2models_normalized(model_3_scatterplot_normed, model_3_data_normalized, 3, parameters_dict_linear_normalized[3], parameters_dict_powerlaw_normalized[3])

# %% [markdown]
# #### Model 4

# %%

domain_space = pd.Series(np.linspace(0.0,0.9,100))
model_4_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_4_linear(x, parameters_dict_linear_normalized[4][0], parameters_dict_linear_normalized[4][1])),
    'Power law model': domain_space.apply(lambda x: model_4_powerlaw(x, parameters_dict_powerlaw_normalized[4][0], parameters_dict_powerlaw_normalized[4][1])) 
})

linear_model = alt.Chart(model_4_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Glacial retreat area (km²)'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-10000, 10000)))
)

powerlaw_model = alt.Chart(model_4_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_4_scatterplot_normed = (linear_model + glacialretreat_normalized)

model_4_scatterplot_normed = add_props_2models_normalized(model_4_scatterplot_normed, model_4_data_normalized, 4, parameters_dict_linear_normalized[4], parameters_dict_powerlaw_normalized[4])

# %% [markdown]
# #### Model 5 

# %%

domain_space = pd.Series(np.linspace(0.0,1.0,100))
model_5_df = pd.DataFrame({
    'domain': domain_space,
    'Linear model': domain_space.apply(lambda x: model_5_linear(x, parameters_dict_linear_normalized[5][0], parameters_dict_linear_normalized[5][1])),
    'Power law model': domain_space.apply(lambda x: model_5_powerlaw(x, parameters_dict_powerlaw_normalized[5][0], parameters_dict_powerlaw_normalized[5][1])) 
})

linear_model = alt.Chart(model_5_df).mark_line(color='grey', strokeWidth=2, opacity=0.5).encode(
    alt.X('domain', title='Nonigneous fraction'),
    alt.Y('Linear model', scale=alt.Scale(domain=(-10000, 10000)), title='')
)

powerlaw_model = alt.Chart(model_5_df).mark_line(color='black', strokeWidth=2, opacity=1).encode(
    alt.X('domain'),
    alt.Y('Power law model')
)

model_5_scatterplot_normed = linear_model + lithology_normalized

model_5_scatterplot_normed = add_props_2models_normalized(model_5_scatterplot_normed, model_5_data_normalized, 5, parameters_dict_linear_normalized[5], parameters_dict_powerlaw_normalized[5])


# %%
def y_label_none(plot):
    return plot.encode(alt.Y(title="", axis=alt.Axis(labels=False)))


(
    (model_1_scatterplot.encode(alt.Y(title="Sediment Yield (ton/yr)")) | y_label_none(model_2_scatterplot) | y_label_none(model_3_scatterplot) | y_label_none(model_4_scatterplot) | y_label_none(model_5_scatterplot))
    &
    (model_1_scatterplot_normed.encode(alt.Y(title="Specific Sediment Yield (ton/km²/yr)")) | y_label_none(model_2_scatterplot_normed) | y_label_none(model_3_scatterplot_normed) | y_label_none(model_4_scatterplot_normed) | y_label_none(model_5_scatterplot_normed))
).configure_legend(
    titleFontSize=22, labelFontSize=22, orient='right'
).configure_axis(
    labelFontSize=16, titleFontSize=16
).configure_title(
    fontSize=22
)


# %%
def y_label_none(plot):
    return plot.encode(alt.Y(title="", axis=alt.Axis(labels=False)))
def no_title(plot):
    return plot.properties(title="")

(
    (no_title(model_1_scatterplot).encode(alt.Y(title="Sediment Yield (ton/yr)")) | y_label_none(no_title(model_2_scatterplot)) | y_label_none(no_title(model_3_scatterplot)) | y_label_none(no_title(model_4_scatterplot)) | y_label_none(no_title(model_5_scatterplot)))
    &
    (no_title(model_1_scatterplot_normed).encode(alt.Y(title="Specific Sediment Yield (ton/km²/yr)")) | y_label_none(no_title(model_2_scatterplot_normed)) | y_label_none(no_title(model_3_scatterplot_normed)) | y_label_none(no_title(model_4_scatterplot_normed)) | y_label_none(no_title(model_5_scatterplot_normed)))
).configure_legend(
    titleFontSize=22, labelFontSize=22, orient='top'
).configure_axis(
    labelFontSize=18, titleFontSize=18
).configure_title(
    fontSize=22
)


# %%
def y_label_none(plot):
    return plot.encode(alt.Y(title="", axis=alt.Axis(labels=False)))


(
    (model_1_scatterplot.properties(title="").encode(alt.Y(title="Sediment Yield (ton/yr)")) | y_label_none(model_2_scatterplot.properties(title="")) | y_label_none(model_3_scatterplot.properties(title="")) | y_label_none(model_4_scatterplot.properties(title="")) | y_label_none(model_5_scatterplot.properties(title="")))
    &
    (darea_normalized.encode(alt.Y(title="Specific Sediment Yield (ton/km²/yr)")) | y_label_none(channelslope_normalized) | y_label_none(hillslope_normalized) | y_label_none(glacialretreat_normalized) | y_label_none(lithology_normalized))
).configure_legend(
    titleFontSize=22, labelFontSize=22, orient='top'
).configure_axis(
    labelFontSize=16, titleFontSize=16
).configure_title(
    fontSize=22
)

# %% [markdown]
# # Save results to CSV

# %%
net_measurements.to_csv("outputs/power_law_data_new.csv")
