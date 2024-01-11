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

if __name__ == "__main__":   

	# %%
	BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
	print(f"retrieved base path: {BASE_PATH}")

	# %% [markdown]
	# # Read file lists

	# %% [markdown]
	# ## Xdem outputs

	# %%
	cum_files = glob.glob('outputs/cum_dv_df/*.pickle') ###
	largerarea_cum_files = glob.glob('outputs/larger_area/cum_dv_df/*.pickle') ###
	bounding_cum_files = glob.glob('outputs/bounding_cum_dv_df/*.pickle') ###
	threshold_pos_files = glob.glob('outputs/threshold_pos_dv_df/*.pickle') 
	threshold_neg_files = glob.glob('outputs/threshold_neg_dv_df/*.pickle')
	hillslope_threshold_pos_files = glob.glob('outputs/hillslope_threshold_pos_dv_df/*.pickle') 
	hillslope_threshold_neg_files = glob.glob('outputs/hillslope_threshold_neg_dv_df/*.pickle')
	fluvial_threshold_pos_files = glob.glob('outputs/fluvial_threshold_pos_dv_df/*.pickle') 
	fluvial_threshold_neg_files = glob.glob('outputs/fluvial_threshold_neg_dv_df/*.pickle')

	cum_process_files = glob.glob('outputs/cum_dv_df_process/*.pickle')
	cum_process_bounding_files = glob.glob('outputs/cum_dv_df_process_bounding/*.pickle')
	# dv_df_process_sums_process/
	process_threshold_neg_files = glob.glob('outputs/threshold_neg_dv_df_process_sums_process/*.pickle')
	process_threshold_pos_files = glob.glob('outputs/threshold_pos_dv_df_process_sums_process/*.pickle')

	process_sums_files = glob.glob("outputs/dv_df_process_sums_process/*.pickle")

	# %% [markdown]
	# ## Xsection outputs

	# %%
	slope_halfkm_files = glob.glob('outputs/slope_grouped_halfkm/*.pickle')
	slope_km_files = glob.glob('outputs/slope_grouped_km/*.pickle')
	elevation_files = glob.glob('outputs/elevation_profiles/*.pickle')

	# %% [markdown]
	# # Load datasets

	# %% [markdown]
	# ## Xdem outputs

	# %% [markdown]
	# ### Gross (pos and neg) datasets thresholded

	# %%
	threshold_pos_df = pd.concat(
	    [pd.read_pickle(f) for f in threshold_pos_files]
	)
	threshold_pos_df['type'] = "positive"
	threshold_neg_df = pd.concat(
	    [pd.read_pickle(f) for f in threshold_neg_files]
	)
	threshold_neg_df['type'] = "negative"
	gross_data_df = pd.concat([threshold_neg_df, threshold_pos_df])

	# %% [markdown]
	# ### Cumulative net and bounding datasets

	# %%

	cum_df = pd.concat(
	    [pd.read_pickle(f) for f in cum_files]
	)

	bounding_cum_df = pd.concat(
	    [pd.read_pickle(f) for f in bounding_cum_files]
	)

	bounding_cum_df['bounding'] = True
	cum_df['bounding'] = False

	cum_and_bounding_cum_df = pd.concat([bounding_cum_df, cum_df])

	# %% [markdown]
	# ### Cumulative net dataset for larger area

	# %%

	largerarea_cum_df = pd.concat(
	    [pd.read_pickle(f) for f in largerarea_cum_files]
	)

	# %% [markdown]
	# ### Gross (pos and neg) datasets thresholded, by erosion type

	# %%
	hillslope_threshold_pos_df = pd.concat([pd.read_pickle(f) for f in hillslope_threshold_pos_files])
	hillslope_threshold_pos_df['type'] = "positive"
	hillslope_threshold_pos_df['process'] = "hillslope"

	hillslope_threshold_neg_df = pd.concat([pd.read_pickle(f) for f in hillslope_threshold_neg_files])
	hillslope_threshold_neg_df['type'] = "negative"
	hillslope_threshold_neg_df['process'] = "hillslope"

	fluvial_threshold_pos_df = pd.concat([pd.read_pickle(f) for f in fluvial_threshold_pos_files])
	fluvial_threshold_pos_df['type'] = "positive"
	fluvial_threshold_pos_df['process'] = "fluvial"

	fluvial_threshold_neg_df = pd.concat([pd.read_pickle(f) for f in fluvial_threshold_neg_files])
	fluvial_threshold_neg_df['type'] = "negative"
	fluvial_threshold_neg_df['process'] = "fluvial"


	gross_data_bytype_df = pd.concat([hillslope_threshold_pos_df, hillslope_threshold_neg_df, fluvial_threshold_pos_df, fluvial_threshold_neg_df])

	# %% [markdown]
	# ### Convert "Annual Mass Wasted" into 1000s of cubic meters

	# %%
	gross_data_df["Annual Mass Wasted"] = gross_data_df["Annual Mass Wasted"]/1000
	gross_data_df["Upper CI"] = gross_data_df["Upper CI"]/1000
	gross_data_df["Lower CI"] = gross_data_df["Lower CI"]/1000

	cum_and_bounding_cum_df["volume"] = cum_and_bounding_cum_df["volume"]/1000
	cum_and_bounding_cum_df["cumulative volume"] = cum_and_bounding_cum_df["cumulative volume"]/1000
	cum_and_bounding_cum_df["Upper CI"] = cum_and_bounding_cum_df["Upper CI"]/1000
	cum_and_bounding_cum_df["Lower CI"] = cum_and_bounding_cum_df["Lower CI"]/1000

	largerarea_cum_df["volume"] = largerarea_cum_df["volume"]/1000
	largerarea_cum_df["cumulative volume"] = largerarea_cum_df["cumulative volume"]/1000
	largerarea_cum_df["Upper CI"] = largerarea_cum_df["Upper CI"]/1000
	largerarea_cum_df["Lower CI"] = largerarea_cum_df["Lower CI"]/1000

	gross_data_bytype_df["Annual Mass Wasted"] = gross_data_bytype_df["Annual Mass Wasted"]/1000
	gross_data_bytype_df["Upper CI"] = gross_data_bytype_df["Upper CI"]/1000
	gross_data_bytype_df["Lower CI"] = gross_data_bytype_df["Lower CI"]/1000

	# %% [markdown]
	# ## Xsection outputs

	# %%
	slope_halfkm_df = pd.concat(
	    [pd.read_pickle(f) for f in slope_halfkm_files]
	)
	slope_km_df = pd.concat(
	    [pd.read_pickle(f) for f in slope_km_files]
	)
	elevation_df = pd.concat(
	    [pd.read_pickle(f) for f in elevation_files]
	)

	# %% [markdown]
	# # Gross erosion/accumulation plots

	# %%
	bars = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	    alt.Color('type',
	        scale=alt.Scale(
	            domain=['negative', 'positive'],
	            range=['#d62728', '#1f77b4']
	        )
	    )
	)

	error_bars = alt.Chart().mark_bar(
	    color="black",
	    width=2
	).encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=""),
	    alt.Y2("Upper CI")
	)

	alt.layer(
	    bars, 
	    error_bars.transform_filter(alt.datum.type == 'negative'), 
	    error_bars.transform_filter(alt.datum.type == 'positive'),
	    data=gross_data_df.drop(columns=['index'])
	).properties(
	    height=100
	).facet(
	    row=alt.Row(
	        'valley:N', 
	        header=alt.Header(labelOrient='top',labelFontWeight="bold"),
	        title="Annualized rate of volumetric change, in 1,000s of m³/yr"
	    )
	).resolve_scale(y='shared')

	# %% [markdown]
	# # Cumulative net erosion plots

	# %%
	cum_plot = alt.Chart().mark_line(point=True).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q'),
	    alt.Color("valley:N")
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr",),
	    alt.Y2("Upper CI"),
	    alt.Color("valley:N")
	)

	alt.layer(
	    cum_plot,
	    error_bars,
	    data=cum_df.drop(columns='index')
	).properties(
	    # height=100
	)

	# %% [markdown]
	# # Cumulative net erosion with 2015 data point

	# %%
	cum_and_bounding_cum_df['larger_area'] = False
	largerarea_cum_df['larger_area'] = True
	largerarea_cum_df['bounding'] = False

	cum_and_bounding_cum_w_largerarea_df = pd.concat([cum_and_bounding_cum_df, largerarea_cum_df])

	# %%
	from datetime import timedelta

	# %%
	cum_and_bounding_cum_w_largerarea_df.loc[
	    cum_and_bounding_cum_w_largerarea_df['bounding'],
	    'end_time'
	] = cum_and_bounding_cum_w_largerarea_df.loc[
	    cum_and_bounding_cum_w_largerarea_df['bounding'],
	    'end_time'
	].apply(lambda date: date + timedelta(days=500))

	# %%
	cum_plot = alt.Chart().mark_line(point=True).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q')
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == False)
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title=""),
	    alt.Y2("Upper CI")
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == False)
	)

	largerarea_cum_plot = alt.Chart().mark_line(
	    point=alt.OverlayMarkDef(color="grey", opacity=0.5),
	    color='grey',
	    opacity=0.5
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q')
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == True)
	)

	largerarea_error_bars = alt.Chart().mark_bar(
	    width=6,
	    color='grey',
	    opacity=0.5
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr"),
	    alt.Y2("Upper CI")
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == True)
	)

	bounding_point = alt.Chart().mark_square(shape='triangle', color='red', size=50).encode(
	    alt.X("end_time:T"),
	    alt.Y("volume:Q"),
	).transform_filter(
	    (alt.datum.bounding == True) & (alt.datum.larger_area == False)
	)

	bounding_point_error_bars = alt.Chart().mark_bar(
	    color="red",
	    width=2
	).encode(
	    alt.X("end_time:T", title=""),
	    alt.Y("Lower CI"),
	    alt.Y2("Upper CI")
	).transform_filter(
	    (alt.datum.bounding == True) & (alt.datum.larger_area == False)
	)

	y_chart = alt.layer(
	    largerarea_cum_plot,
	    largerarea_error_bars,
	    bounding_point,
	    bounding_point_error_bars,
	    cum_plot,
	    error_bars,
	    data=cum_and_bounding_cum_w_largerarea_df.drop(columns='index')
	).properties(
	    width=300, height=100
	).facet(
	    column=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labelOrient='top',
	            labelFontWeight="bold",
	            # labelPadding=-10
	        ),
	        title="Cumulative net change, in 1,000s of m³/yr",  
	        
	    ),
	    spacing=1
	).resolve_scale(
	    y='independent'
	)

	y_chart

	# %%
	cum_plot = alt.Chart().mark_line(point=True).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q')
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == False)
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr"),
	    alt.Y2("Upper CI")
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == False)
	)

	largerarea_cum_plot = alt.Chart().mark_line(
	    point=alt.OverlayMarkDef(color="grey", opacity=0.5),
	    color='grey',
	    opacity=0.5
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q')
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == True)
	)

	largerarea_error_bars = alt.Chart().mark_bar(
	    width=6,
	    color='grey',
	    opacity=0.5
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title=""),
	    alt.Y2("Upper CI")
	).transform_filter(
	    (alt.datum.bounding == False) & (alt.datum.larger_area == True)
	)

	alt.layer(
	    largerarea_cum_plot,
	    largerarea_error_bars,
	    cum_plot,
	    error_bars,
	    data=cum_and_bounding_cum_w_largerarea_df.drop(columns='index')
	).properties(
	    width=300, height=100
	).facet(
	    column=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labelOrient='top',
	            labelFontWeight="bold",
	            # labelPadding=-10
	        ),
	        title="Cumulative net change, in 1,000s of m³/yr",
	        
	    ),
	    spacing=1
	).resolve_scale(
	    y='independent'
	)

	# %%
	cum_plot = alt.Chart().mark_line(point=True).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q')
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr"),
	    alt.Y2("Upper CI")
	)

	simple_cum_chart = alt.layer(
	    cum_plot.transform_filter((alt.datum.bounding == False) & (alt.datum.larger_area == False)),
	    error_bars.transform_filter((alt.datum.bounding == False) & (alt.datum.larger_area == False)),
	    data=cum_and_bounding_cum_w_largerarea_df.drop(columns='index')
	).properties(
	    width=300, height=100
	).facet(
	    column=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labelOrient='top',
	            labelFontWeight="bold",
	            # labelPadding=-10
	        ),
	        title="Cumulative net change, in 1,000s of m³/yr",
	        
	    ),
	    # spacing=1
	).resolve_scale(
	    y='independent'
	)

	simple_cum_chart

	# %% [markdown]
	# # Parse Lithology

	# %% [markdown]
	# ## With erosion polygons

	# %%
	terrain_attrs_erosionarea = pd.read_csv("outputs/terrain_attributes_erosionarea.csv")
	terrain_attrs_erosionarea = terrain_attrs_erosionarea.rename(columns={'name': 'Valley Name'})
	terrain_attrs_erosionarea['drainage area (km)'] = terrain_attrs_erosionarea['drainage area'] / 1e6

	# %%
	bareground_polys_gdf = gpd.GeoDataFrame(
	    terrain_attrs_erosionarea,
	    geometry = terrain_attrs_erosionarea['geometry'].apply(wkt.loads),
	    crs='EPSG:32610'
	)

	# %%
	erosion_polys_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/**/erosion.geojson"), recursive=True)
	erosion_gdf = pd.concat([gpd.read_file(f) for f in erosion_polys_fns])

	gully_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/gully.shp")
	wasting_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/mass_wasting.shp")
	glacial_debutressing_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/debutressing.shp")

	erosion_polygons_gdf = pd.concat([
	    gpd.read_file(gully_polygons_fn),
	    gpd.read_file(wasting_polygons_fn),
	    gpd.read_file(glacial_debutressing_polygons_fn)
	])

	# %%
	import fiona
	lithology_gdf = gpd.read_file(os.path.join(BASE_PATH, "geology/q111shp_cropped_mtbaker/gunit_polygon.shp"))

	# %%
	lithology_gdf.plot(column='GUNIT_LABE')

	# %%
	bareground_lithology_by_valley = bareground_polys_gdf.groupby("Valley Name").apply(lithology_gdf.clip).reset_index()
	bareground_lithology_by_valley['AREA'] = bareground_lithology_by_valley.geometry.area
	bareground_lithology_by_valley = bareground_lithology_by_valley.query("GUNIT_LABE != 'ice'")


	erosion_lithology_by_valley = erosion_polygons_gdf.groupby('name').apply(lithology_gdf.clip).reset_index()

	erosion_lithology_by_valley['AREA'] = erosion_lithology_by_valley.geometry.area
	erosion_lithology_by_valley = erosion_lithology_by_valley.query("GUNIT_LABE != 'ice'")

	# %% [markdown]
	# create lithology dataset for each valley and combine datasets

	# %%
	# gunit_convert = {
	#     'Qat': "Pleistocene,at,alpine glacial till,loose material",
	#     'Qva(b)': "Holocene-Pleistocene,va,andesite flows,igneous",
	#     'KJm(n1)': "Cretaceous-Jurassic,m,marine sedimentary rocks,sedimentary",
	#     'Qad': "Pleistocene,ad,alpine glacial drift,loose material",
	#     'Kigb': "Cretaceous,igb,gabbro,volcanic",
	#     'JPMhmc(b)': "Jurassic-Permian,hmc,heterogeneous metamorphic rocks,metamorphic",
	#     'Qvx(b)': "Quaternary,vx,volcanic breccia,volcanic",
	#     'KJm(n2)': "Cretaceous-Jurassic,m,marine sedimentary rocks,sedimentary",
	#     'Qls': "Quaternary,ls,mass-wasting deposits,loose material",
	#     'Migd': "Miocene,igd,granodiorite,igneous",
	#     'Qta': "Holocene-Pleistocene,ta,talus deposits,loose material",
	#     'Qva(p)': "Quaternary,va,andesite flows,igneous",
	#     'Qvt(ks)': "Pleistocene,vt,tuffs and tuff breccias,volcanic",
	#     'KJm(n4)': "Cretaceous-Jurassic,m,marine sedimentary rocks,sedimentary",
	#     'Qva(ld)': "Quaternary,va,andesite flows,igneous",
	#     'PMvb(c)': "Permian,vb,basalt flows,igneous",
	#     'Qva(bb)': "Quaternary,va,andesite flows,igneous",
	# }

	gunit_convert = {
	    'Qat': "Pleistocene,at,alpine glacial till,Pleistocene glaciogenic material",
	    'Qva(b)': "Holocene-Pleistocene,va,andesite flows,Igneous",
	    'KJm(n1)': "Cretaceous-Jurassic,m,marine sedimentary rocks,Sedimentary",
	    'Qad': "Pleistocene,ad,alpine glacial drift,Pleistocene glaciogenic material",
	    'Kigb': "Cretaceous,igb,gabbro,Igneous",
	    'JPMhmc(b)': "Jurassic-Permian,hmc,heterogeneous metamorphic rocks,Metamorphic",
	    'Qvx(b)': "Quaternary,vx,volcanic breccia,Volcanic",
	    'KJm(n2)': "Cretaceous-Jurassic,m,marine sedimentary rocks,Sedimentary",
	    'Qls': "Quaternary,ls,mass-wasting deposits,Quaternary mass wasting deposits",
	    'Migd': "Miocene,igd,granodiorite,Igneous",
	    'Qta': "Holocene-Pleistocene,ta,talus deposits,Holocene-pleistocene talus",
	    'Qva(p)': "Quaternary,va,andesite flows,Igneous",
	    'Qvt(ks)': "Pleistocene,vt,tuffs and tuff breccias,Volcanic",
	    'KJm(n4)': "Cretaceous-Jurassic,m,marine sedimentary rocks,Sedimentary",
	    'Qva(ld)': "Quaternary,va,andesite flows,Igneous",
	    'PMvb(c)': "Permian,vb,basalt flows,Igneous",
	    'Qva(bb)': "Quaternary,va,andesite flows,Igneous",
	}

	# %%
	bareground_lithology_by_valley['unit description'] = bareground_lithology_by_valley['GUNIT_LABE'].apply(gunit_convert.get)
	bareground_lithology_by_valley['description'] = bareground_lithology_by_valley['unit description'].apply(lambda s: s.split(",")[-1])

	erosion_lithology_by_valley['unit description'] = erosion_lithology_by_valley['GUNIT_LABE'].apply(gunit_convert.get)
	erosion_lithology_by_valley['description'] = erosion_lithology_by_valley['unit description'].apply(lambda s: s.split(",")[-1])

	# %% [markdown]
	# ### Plot proportion of each lithology by area in each valley

	# %%
	src = erosion_lithology_by_valley.copy()
	# normalize area of of each unit by total area measured in each valley
	src['AREA'] = src['AREA'] / src.groupby('name')['AREA'].transform('sum')

	alt.Chart(src).mark_arc().encode(
	    alt.Theta("AREA:Q"),
	    alt.Color("unit description:N"),
	    alt.Facet("name:N", columns=5)
	).properties(width=100, height=100).configure_view(strokeWidth=0).configure_legend(labelLimit=0)

	# %%
	alt.Chart(
	    pd.DataFrame(bareground_lithology_by_valley.groupby(['description', 'unit description'])['AREA'].sum()).reset_index()
	).mark_bar().encode(
	    alt.Y("unit description:N", sort='x', axis=alt.Axis(labelLimit=500, title='')),
	    alt.X("AREA:Q", title='Area (m²)'),
	    alt.Color("description:N")
	).properties(
	    title = 'Total area covered by surface lithology units in measured bareground area in 10 watersheds'
	)

	# %%
	least_common_5_units = bareground_lithology_by_valley.groupby(['description', 'unit description'])['AREA'].sum().sort_values().reset_index()['unit description'].head(5)

	least_common_5_units

	# %%
	src = bareground_lithology_by_valley.copy()

	# REMOVE LEAST COMMON 5 UNITS
	src = src[~src['unit description'].isin(least_common_5_units)]

	# normalize area of of each unit by total area measured in each valley
	src['AREA'] = src['AREA'] / src.groupby('Valley Name')['AREA'].transform('sum')

	alt.Chart(src).mark_arc().encode(
	    alt.Theta("AREA:Q"),
	    alt.Color("unit description:N"),
	    alt.Facet("Valley Name:N", columns=5)
	).properties(
	    width=100, height=100,
	    title = 'Relative prevalence of surface lithology units in measured bareground area'
	).configure_view(strokeWidth=0).configure_legend(labelLimit=0)

	# %%
	bareground_lithology_by_valley['description'].unique()
	res = bareground_lithology_by_valley.groupby(["Valley Name", "description"])[['AREA']].sum()
	denom = res.groupby('Valley Name')['AREA'].sum()
	res['AREA'] = res['AREA'] / denom
	res = res.reset_index()
	alt.Chart(
	    res
	).mark_arc().encode(
	    theta=alt.Theta(field="AREA", type="quantitative"),
	    color=alt.Color(field="description", type="nominal"),
	    facet=alt.Facet("Valley Name:N", columns=5),
	    # groupby = "Valley Name:N"
	).properties(width=100, height=100).configure_view(strokeWidth=0)

	# %%
	lithology_data =  res.copy()

	# %% [markdown]
	# ## With process polygons

	# %%
	process_polys_gdf = gpd.GeoDataFrame(
	    pd.read_csv("outputs/terrain_attributes_processpolygons.csv"),
	    geometry = pd.read_csv("outputs/terrain_attributes_processpolygons.csv")['geometry'].apply(wkt.loads),
	    crs='EPSG:32610'
	)

	# %%
	process_polys_gdf.head(3)

	# %%
	process_polys_gdf.geometry.type.unique()


	# %%
	def process_geometry(row):
	    res = lithology_gdf.clip(row.geometry)
	    res['AREA'] = res.geometry.area
	    res['volume'] = res['AREA']*row['ddem']
	    res['type'] = row['type']
	    return res

	results = process_polys_gdf.apply(process_geometry, axis=1)
	lithology_by_process = pd.concat(list(results))
	lithology_by_process = lithology_by_process.query("GUNIT_LABE != 'ice'")
	lithology_by_process['unit description'] = lithology_by_process['GUNIT_LABE'].apply(gunit_convert.get)
	lithology_by_process['description'] = lithology_by_process['unit description'].apply(lambda s: s.split(",")[-1])
	lithology_by_process = lithology_by_process.reset_index()

	# %% [markdown]
	# ### Plot proportion of each lithology by eroded volume, organized by process typem

	# %%
	res = lithology_by_process.groupby(["type", "description"])[['volume']].sum()
	denom = res.groupby('type')['volume'].sum()
	res['volume'] = res['volume'] / denom
	res = res.reset_index()
	alt.Chart(
	    res
	).mark_arc().encode(
	    theta=alt.Theta(field="volume", type="quantitative"),
	    color=alt.Color(field="description", type="nominal"),
	    facet=alt.Facet("type:N", columns=5),
	    # groupby = "Valley Name:N"
	).properties(width=100, height=100).configure_view(strokeWidth=0)

	# %%
	### Plot proportion of each lithology by area, organized by process type

	# %%
	res = lithology_by_process.groupby(["type", "description"])[['AREA']].sum()
	denom = res.groupby('type')['AREA'].sum()
	res['AREA'] = res['AREA'] / denom
	res = res.reset_index()
	alt.Chart(
	    res
	).mark_arc().encode(
	    theta=alt.Theta(field="AREA", type="quantitative"),
	    color=alt.Color(field="description", type="nominal"),
	    facet=alt.Facet("type:N", columns=5),
	    # groupby = "Valley Name:N"
	).properties(width=100, height=100).configure_view(strokeWidth=0)

	# %% [markdown]
	# # Gross erosion/accumulation plots by process

	# %%
	bars = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	    alt.Color('type',
	        scale=alt.Scale(
	            domain=['negative', 'positive'],
	            range=['#d62728', '#1f77b4']
	        )
	    )
	)

	error_bars = alt.Chart().mark_bar(
	    color="black",
	    width=2
	).encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=""),
	    alt.Y2("Upper CI")
	)

	alt.layer(
	    bars, 
	    error_bars.transform_filter(alt.datum.type == 'negative'), 
	    error_bars.transform_filter(alt.datum.type == 'positive'),
	    data=gross_data_bytype_df.drop(columns=['index'])
	).properties(
	    height=200
	).facet(
	    row=alt.Row(
	        'valley:N', 
	        header=alt.Header(
	            # labelOrient='top',
	            labelFontWeight="bold", 
	            labelPadding=0
	        ),
	        title="Annualized rate of volumetric change, in 1,000s of m³/yr"
	    ),
	    column=alt.Column("process:N"),
	    spacing=1
	).resolve_scale(
	    y='independent'
	)

	# %%
	from datetime import timedelta
	gross_data_bytype_df['Average Date Plus'] = gross_data_bytype_df['Average Date'] + timedelta(days=120)
	gross_data_bytype_df['Average Date Minus'] = gross_data_bytype_df['Average Date'] - timedelta(days=120)

	# %%
	bars = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	    alt.Color('type',
	        scale=alt.Scale(
	            domain=['negative', 'positive'],
	            range=['#d62728', '#1f77b4']
	        )
	    )
	)

	error_bars = alt.Chart().encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=""),
	    alt.Y2("Upper CI")
	)

	alt.layer(
	    bars, 
	    error_bars.transform_filter(alt.datum.type == 'negative').mark_bar(color="black", width=2).encode(alt.X('Average Date Minus:T')), 
	    error_bars.transform_filter(alt.datum.type == 'positive').mark_bar(color="black", width=2).encode(alt.X('Average Date Plus:T')),
	    data=gross_data_bytype_df.drop(columns=['index'])
	).properties(
	    height=200
	).facet(
	    row=alt.Row(
	        'valley:N', 
	        header=alt.Header(
	            # labelOrient='top',
	            labelFontWeight="bold", 
	            labelPadding=0
	        ),
	        title="Annualized rate of volumetric change, in 1,000s of m³/yr"
	    ),
	    column=alt.Column("process:N"),
	    spacing=1
	).resolve_scale(
	    y='independent'
	)

	# %%
	bars = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	    alt.Color('type',
	        scale=alt.Scale(
	            domain=['negative', 'positive'],
	            range=['#d62728', '#1f77b4']
	        )
	    )
	)

	error_bars = alt.Chart().mark_bar(
	    color="black",
	    width=2
	).encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=""),
	    alt.Y2("Upper CI")
	)

	layer = alt.layer(
	    bars, 
	    error_bars.transform_filter(alt.datum.type == 'negative'), 
	    error_bars.transform_filter(alt.datum.type == 'positive'),
	    data=gross_data_bytype_df.drop(columns=['index'])
	).properties(
	    height=100
	).facet(
	    row=alt.Row(
	        'valley:N', 
	        header=alt.Header(
	            # labelOrient='top',
	            labelFontWeight="bold", 
	            # labelPadding=-10
	        ),
	        title="Annualized rate of volumetric change, in 1,000s of m³/yr"
	    ),
	    spacing=1
	)

	layer.transform_filter(
	    alt.datum.process == 'fluvial'
	).properties(title='fluvial') | layer.transform_filter(
	    alt.datum.process == 'hillslope'
	).properties(title='hillslope')

	# %% [markdown]
	# # Valley slope plots

	# %%
	# Rainbow: 0 - 72
	# Mazama: 0 - 63
	# Deming: 0 - 69
	# Coleman: 0 - ...
	# Easton: 

	# %%
	df = elevation_df.query("valley == 'Rainbow'").query(
	    "n_from_glacial_max > 0"
	).query(
	    "n_from_glacial_max < 72"
	).sort_values(
	    "n_from_glacial_max"
	).query("time == '2015_09_01'")

	np.degrees(np.arctan(
	    (df.iloc[-1]['elevation'] - df.iloc[0]['elevation']) / (
	        df.iloc[-1]['path_distance_from_glacier'] - df.iloc[0]['path_distance_from_glacier']
	    )
	))

	# %%
	df = elevation_df.query("valley == 'Mazama'").query(
	    "n_from_glacial_max > 0"
	).query(
	    "n_from_glacial_max < 63"
	).sort_values(
	    "n_from_glacial_max"
	).query("time == '2015_09_01'")

	np.degrees(np.arctan(
	    (df.iloc[-1]['elevation'] - df.iloc[0]['elevation']) / (
	        df.iloc[-1]['path_distance_from_glacier'] - df.iloc[0]['path_distance_from_glacier']
	    )
	))

	# %%
	df = elevation_df.query("valley == 'Deming'").query(
	    "n_from_glacial_max > 0"
	).query(
	    "n_from_glacial_max < 69"
	).sort_values(
	    "n_from_glacial_max"
	).query("time == '2015_09_01'")

	np.degrees(np.arctan(
	    (df.iloc[-1]['elevation'] - df.iloc[0]['elevation']) / (
	        df.iloc[-1]['path_distance_from_glacier'] - df.iloc[0]['path_distance_from_glacier']
	    )
	))

	# %%
	df = elevation_df.query("valley == 'Coleman'").query(
	    "n_from_glacial_max > 0"
	).sort_values(
	    "n_from_glacial_max"
	).query("time == '2015_09_01'")

	np.degrees(np.arctan(
	    (df.iloc[-1]['elevation'] - df.iloc[0]['elevation']) / (
	        df.iloc[-1]['path_distance_from_glacier'] - df.iloc[0]['path_distance_from_glacier']
	    )
	))

	# %%
	alt.Chart(slope_halfkm_df).mark_line(point=True).encode(
	    alt.X('time:T', title=""),
	    alt.Y('slope:Q', title="Valley floor slope"),
	    alt.Color("valley:N"),
	    alt.Facet('Half kilometer downstream from glacier:O', title='Half kilometer downstream from glacier')
	).properties(width=200)


	# %%
	alt.Chart(slope_km_df).mark_line(point=True).encode(
	    alt.X('time:T', title=""),
	    alt.Y('slope:Q', title="Valley floor slope"),
	    alt.Color("valley:N"),
	    alt.Facet('Kilometer downstream from glacier:O', title='Kilometer downstream from glacier')
	).properties(width=200)

	# %%
	elevation_df.time.unique()

	# %%
	elevation_df[elevation_df['time'].isin(['1947_09_14', '1970_09_29', '1977_09_27', '1979_10_06', '2015_09_01'])].time.unique()

	# %%
	elevation_df.query("valley =='Rainbow'").time.unique()

	# %%
	elevation_df.query("valley =='Mazama'").time.unique()

	# %%
	elevation_df.query("valley =='Deming'").time.unique()

	# %%
	elevation_df = elevation_df[elevation_df['time'].isin(['1947_09_14', '1970_09_29', '1977_09_27', '1979_10_06', '2015_09_01'])]

	# %%
	elevation_df = elevation_df[elevation_df['time'].isin(['1947_09_14', '1970_09_29', '1977_09_27', '1979_10_06', '2015_09_01'])]
	# elevation_df = elevation_df[~(
	#     (elevation_df['time'] == '1979_10_06') & (elevation_df['valley'] == 'Rainbow')
	# )]
	# elevation_df = elevation_df[~(
	#     (elevation_df['time'] == '1979_10_06') & (elevation_df['valley'] == 'Mazama')
	# )]
	elevation_df

	# %%
	src = elevation_df[[ "time", "path_distance_from_glacier", "elevation", "glacial", "valley"]].reset_index()
	alt.Chart(
	    src
	).mark_line(
	    strokeWidth=1, clip=True
	).transform_filter(
	    {'field': 'valley', 'oneOf': ['Deming', 'Mazama', 'Rainbow']}
	).transform_filter(
	    alt.datum.glacial==False
	).encode(
	    alt.X(
	        "path_distance_from_glacier:Q", 
	        title="Distance downstream from observed glacial maximum", 
	        scale=alt.Scale(domain=[-800, 1800])
	    ),
	    alt.Y(
	        "elevation:Q", 
	        scale=alt.Scale(domain=[950, 1200]), 
	        title="Valley floor elevation, in meters"
	    ),
	    alt.StrokeDash("time:O", scale=alt.Scale(
	            domain = ['1947_09_14', '1970_09_29', '1977_09_27', '1979_10_06', '2015_09_01'],
	            range= [[6,1.5], [4, 2.5], [1, 2],[1, 2], [1, 0]]
	        )
	    ),
	    alt.Color("valley:N"),
	).properties(
	    width = 500,
	    height = 500,
	    title='Elevation profile of main stream channel downstream of glaciers'
	).configure_axis(grid=False)

	# %%
	src = elevation_df[[ "time", "path_distance_from_glacier", "elevation", "glacial", "valley"]].reset_index()
	src.loc
	src.loc[src.valley == 'Rainbow', 'elevation'] = src.loc[src.valley == 'Rainbow', 'elevation'] + 30
	src.loc[src.valley == 'Deming', 'elevation'] = src.loc[src.valley == 'Deming', 'elevation'] + 10
	alt.Chart(
	    src
	).mark_line(
	    strokeWidth=1, clip=True
	).transform_filter(
	    {'field': 'valley', 'oneOf': ['Deming', 'Mazama', 'Rainbow']}
	).transform_filter(
	    alt.datum.glacial==False
	).encode(
	    alt.X(
	        "path_distance_from_glacier:Q", 
	        title="Distance downstream from observed glacial maximum", 
	        scale=alt.Scale(domain=[-400, 1900])
	    ),
	    alt.Y(
	        "elevation:Q", 
	        scale=alt.Scale(domain=[950, 1200]), 
	        title="Valley floor elevation, in meters"
	    ),
	    alt.StrokeDash("time:O", scale=alt.Scale(
	            domain = ['1947_09_14', '1970_09_29', '1977_09_27', '1979_10_06', '2015_09_01'],
	            range= [[8,1.5], [4, 2.5], [1, 2],[1, 2], [1, 0]]
	        )
	    ),
	    alt.Color("valley:N"),
	).properties(
	    width = 1000,
	    height = 500,
	    title={
	        "text":'Elevation profile of main stream channel downstream of glaciers',
	        "subtitle": "Rainbow data shifted upwards 30 meters, Deming data shifted upwards 10 meters for clarity."
	    }
	).configure_axis(grid=False)

	# %%
	for date in [
	    '2015_09_01',
	    '1970_09_29',
	    '1947_09_14'
	]:
	    rainbow_slope = src.query("valley == 'Rainbow'").query(f"time == '{date}'").query(
	    "path_distance_from_glacier < 1200").query("path_distance_from_glacier > 0").sort_values("path_distance_from_glacier")

	    print(np.degrees(np.arctan(
	    (rainbow_slope['elevation'].iloc[-1] - rainbow_slope['elevation'].iloc[0]) / (
	        rainbow_slope['path_distance_from_glacier'].iloc[-1] - rainbow_slope['path_distance_from_glacier'].iloc[0]
	    )
	    )))

	# %%
	for date in [
	    '2015_09_01',
	    '1970_09_29',
	    '1947_09_14'
	]:
	    mazama_slope = src.query("valley == 'Mazama'").query(f"time == '{date}'").query(
	        "path_distance_from_glacier < 400").query("path_distance_from_glacier > 0").sort_values("path_distance_from_glacier")

	    print(np.degrees(np.arctan(
	    (mazama_slope['elevation'].iloc[-1] - mazama_slope['elevation'].iloc[0]) / (
	        mazama_slope['path_distance_from_glacier'].iloc[-1] - mazama_slope['path_distance_from_glacier'].iloc[0]
	    )
	    )))

	# %%
	for date in [
	    '2015_09_01',
	    '1979_10_06',
	    '1970_09_29',
	    '1947_09_14'
	]:
	    deming_slope = src.query("valley == 'Deming'").query(f"time == '{date}'").query(
	        "path_distance_from_glacier < 1400").query("path_distance_from_glacier > 0").sort_values("path_distance_from_glacier")
	    print(
	    np.degrees(np.arctan(
	    (deming_slope['elevation'].iloc[-1] - deming_slope['elevation'].iloc[0]) / (
	        deming_slope['path_distance_from_glacier'].iloc[-1] - deming_slope['path_distance_from_glacier'].iloc[0]
	    )
	    )))

	# %% [markdown]
	# # Cumulative net erosion plots by process

	# %%
	cum_process_df = pd.concat(
	    [pd.read_pickle(f) for f in cum_process_files]
	)

	cum_process_df['cumulative volume'] = cum_process_df['cumulative volume']/1000
	cum_process_df['Lower CI'] = cum_process_df['Lower CI']/1000
	cum_process_df['Upper CI'] = cum_process_df['Upper CI']/1000

	cum_process_df['type'] = cum_process_df['type'].apply(lambda existing_type: 'glacial' if existing_type == 'glacial debutressing' else existing_type)
	cum_process_df['bounding'] = False

	cum_process_df.head(3)

	# %%
	cum_process_bounding_df = pd.concat(
	    [pd.read_pickle(f) for f in cum_process_bounding_files]
	)

	cum_process_bounding_df['cumulative volume'] = cum_process_bounding_df['cumulative volume']/1000
	cum_process_bounding_df['Lower CI'] = cum_process_bounding_df['Lower CI']/1000
	cum_process_bounding_df['Upper CI'] = cum_process_bounding_df['Upper CI']/1000

	cum_process_bounding_df['type'] = cum_process_bounding_df['type'].apply(lambda existing_type: 'glacial' if existing_type == 'glacial debutressing' else existing_type)

	cum_process_bounding_df['bounding'] = True

	cum_process_bounding_df = cum_process_bounding_df.dropna(subset='start_time')

	cum_process_bounding_df.head(3)

	# %%
	all_data = cum_and_bounding_cum_w_largerarea_df.query("~bounding").query("~larger_area").query("valley != 'Easton'")[['valley', 'end_time','cumulative volume','Lower CI','Upper CI']]
	all_data['type'] = 'all'
	bytype_data = cum_process_df.query("valley != 'Easton'")
	bytype_bounding_data = cum_process_bounding_df.query("valley != 'Easton'")

	# %%
	cum_data_alltogether = pd.concat([bytype_data, bytype_bounding_data, all_data])[['valley', 'start_time', 'end_time','cumulative volume','type','Lower CI','Upper CI', 'bounding', 'Annual Mass Wasted']]

	# %%
	cum_plot = alt.Chart().mark_line(point=True).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q'),
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr"),
	    alt.Y2("Upper CI")
	)

	simple_cum_chart = alt.layer(
	    cum_plot.transform_filter((alt.datum.bounding == False) & (alt.datum.larger_area == False)),
	    error_bars.transform_filter((alt.datum.bounding == False) & (alt.datum.larger_area == False)),
	    data=cum_and_bounding_cum_w_largerarea_df.drop(columns='index')
	).properties(
	    width=300, height=100
	).facet(
	    column=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labelOrient='top',
	            labelFontWeight="bold",
	            # labelPadding=-10
	        ),
	        title="Cumulative net change, in 1,000s of m³/yr",
	        
	    ),
	    # spacing=1
	).resolve_scale(
	    y='independent'
	)

	simple_cum_chart

	# %%
	src = cum_and_bounding_cum_w_largerarea_df.drop(columns='index')
	src = src[src['Upper CI'] != 0].query("bounding == False")
	src = src.dropna()
	src

	# %%



	src = cum_and_bounding_cum_w_largerarea_df.drop(columns='index')
	src['cumulative volume'] = - src['cumulative volume']/1000

	# domain = ["1940-01-01", "2020-01-01"]
	domain = ["1940", "2020"]

	cum_plot = alt.Chart().mark_line().encode(
	    # alt.X('end_time:T', title='Time', timeUnit='yearmonthdate', scale=alt.Scale(domain=domain)),
	    alt.X('end_time:T', title='Time', timeUnit='year', scale=alt.Scale(domain=domain)),
	    alt.Y('cumulative volume:Q'),
	    alt.Color('valley:N')
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr"),
	    # alt.Color('valley:N')
	)

	simple_cum_chart = alt.layer(
	    cum_plot.transform_filter((alt.datum.bounding == False) & (alt.datum.larger_area == False)),
	    # error_bars.transform_filter((alt.datum.bounding == False) & (alt.datum.larger_area == False)),
	    data=src
	).properties(
	    width=350, height=200
	)

	simple_cum_chart

	# %%
	# scale=alt.Scale(domain=['fluvial', 'hillslope', 'all'], range=['#1f77b4', '#ff7f0e', '#d62728'])

	src = gross_data_bytype_df.drop(columns=['index'])
	src = src.query("valley != 'Easton'")
	# src = src.query("valley != 'Coleman'")

	src['Annual Mass Wasted'] = src['Annual Mass Wasted'].apply(lambda x: 1 if x == 0 else x)
	src['Upper CI'] = src['Upper CI'].apply(lambda x: 1 if x == 0 else x)
	src['Lower CI'] = src['Lower CI'].apply(lambda x: 1 if x == 0 else x)

	src['Annual Mass Wasted'] = - src['Annual Mass Wasted']
	src['Upper CI'] = - src['Upper CI']
	src['Lower CI'] = - src['Lower CI']

	bars_neg = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	).encode(
	    alt.X('start_time:T', axis=alt.Axis(labels=False), title=None),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted', scale=alt.Scale(domain=[-50,100], nice=False)),
	    alt.Color("valley:N")
	)

	bars_pos = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	    opacity=0.5
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	    alt.Color("valley:N")

	)

	error_bars = alt.Chart().mark_bar(
	    color="black",
	    width=2
	).encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=[
	        # "Annualized rate of volumetric change,", "in 1,000s of m³/yr"
	        ]),
	    alt.Y2("Upper CI")
	)

	hillslope_combo_gross_bars_chart = alt.layer(
	    bars_neg.transform_filter(alt.datum.type == 'negative').transform_filter(alt.datum.process == 'hillslope'), 
	    bars_pos.transform_filter(alt.datum.type == 'positive').transform_filter(alt.datum.process == 'hillslope'), 
	    error_bars.transform_filter(alt.datum.type == 'negative').transform_filter(alt.datum.process == 'hillslope'), 
	    error_bars.transform_filter(alt.datum.type == 'positive').transform_filter(alt.datum.process == 'hillslope'),
	    data=src
	).properties(
	    width=350,
	    height=150
	).facet(
	    row=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labels=False,
	            labelFontSize=16
	            # labelOrient='top',
	            # labelFontWeight="bold", 
	            # labelPadding=0
	        ),
	        title=None
	    ),
	    # spacing=1
	)

	hillslope_combo_gross_bars_chart

	# %%
	# (hillslope_combo_gross_bars_chart.transform_filter(alt.datum.valley != 'Coleman') & simple_cum_chart).configure_legend(
	(hillslope_combo_gross_bars_chart.transform_filter(alt.datum.valley != 'Coleman').transform_filter(alt.datum.valley != 'Rainbow')
	 & simple_cum_chart).configure_legend(
	    titleFontSize=18, 
	    labelFontSize=18, 
	    orient='top', 
	    symbolSize=1000, 
	    symbolStrokeWidth=2
	).configure_axis(
	    labelFontSize=18, 
	    titleFontSize=18,
	).resolve_scale(
	    x='shared'
	)

	# %%
	from datetime import timedelta

	# %%
	cum_data_alltogether.loc[cum_data_alltogether['bounding'] == True, 'end_time'] = cum_data_alltogether.loc[cum_data_alltogether['bounding'] == True, 'end_time'].apply(lambda date: date + timedelta(days=1000))

	# %%
	cum_data_alltogether.loc[cum_data_alltogether["type"] == 'all', 'bounding'] = False

	# %%
	src = cum_data_alltogether
	src = src.query("valley != 'Easton'")
	part1 = src[~src['type'].isin(['gully', 'mass wasting'])]
	part2 = src[src['type'].isin(['gully', 'mass wasting'])].groupby(['valley', 'start_time', 'end_time', 'bounding']).sum().reset_index().assign(type='not glacial')
	src = pd.concat([part1, part2])

	src = pd.concat([
	    src,
	    pd.DataFrame({
	        'valley': ['Coleman', 'Deming', 'Mazama', 'Rainbow'],
	        'end_time': [datetime(1947, 9, 14), datetime(1947, 9, 14), datetime(1947, 9, 14), datetime(1947, 9, 14)],
	        'cumulative volume': [0,0,0,0],
	        'bounding': [False, False, False, False],
	        'type': ['not glacial', 'not glacial', 'not glacial', 'not glacial']
	    })
	])

	cum_plot = alt.Chart().mark_line(
	    point={'size':20},
	     strokeWidth=2.5,
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q', 
	        scale=alt.Scale(domain=[-4000, 500])
	    ),
	    # alt.Color("type:N")
	    alt.Color(
	        "type:N", 
	        scale=alt.Scale(domain=['fluvial', 'hillslope', 'all', 'glacial', 'not glacial'], range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c'])
	    ),
	    alt.StrokeDash("type:N")
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", ),
	    alt.Y2("Upper CI"),
	    alt.Color(
	        "type:N", 
	        scale=alt.Scale(domain=['fluvial', 'hillslope', 'all', 'glacial', 'not glacial'], range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c'])
	    )
	)


	cumulative_by_process_chart_no_glacial = alt.layer(
	    cum_plot.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['all', 'fluvial', 'hillslope'])).transform_filter(alt.datum.bounding == False),
	    error_bars.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['all', 'fluvial', 'hillslope'])).transform_filter(alt.datum.bounding == False),
	    # cum_plot2.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['not glacial', 'glacial'])).transform_filter(alt.datum.bounding == False),
	    # error_bars2.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['not glacial', 'glacial'])).transform_filter(alt.datum.bounding == False),
	    # bounding_points.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['fluvial', 'hillslope', 'all'])).transform_filter(alt.datum.bounding == True),
	    # bounding_points2.transform_filter(alt.datum.type == 'glacial').transform_filter(alt.datum.bounding == True),
	    data=src
	).properties(
	    width=350
	).facet(
	    column=alt.Column('valley:N', header=alt.Header(labels=False, labelFontSize=16), title=''),
	)
	cumulative_by_process_chart_no_glacial

	# %%
	src = cum_data_alltogether
	src = src.query("valley != 'Easton'")
	part1 = src[~src['type'].isin(['gully', 'mass wasting'])]
	part2 = src[src['type'].isin(['gully', 'mass wasting'])].groupby(['valley', 'start_time', 'end_time', 'bounding']).sum().reset_index().assign(type='not glacial')
	src = pd.concat([part1, part2])

	src = pd.concat([
	    src,
	    pd.DataFrame({
	        'valley': ['Coleman', 'Deming', 'Mazama', 'Rainbow'],
	        'end_time': [datetime(1947, 9, 14), datetime(1947, 9, 14), datetime(1947, 9, 14), datetime(1947, 9, 14)],
	        'cumulative volume': [0,0,0,0],
	        'bounding': [False, False, False, False],
	        'type': ['not glacial', 'not glacial', 'not glacial', 'not glacial']
	    })
	])

	cum_plot = alt.Chart().mark_line(
	        point={'size':20},
	     strokeWidth=2.5,
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q', 
	        scale=alt.Scale(domain=[-4000, 500])
	    ),
	    # alt.Color("type:N")
	    alt.Color(
	        "type:N", 
	        scale=alt.Scale(domain=['fluvial', 'hillslope', 'all', 'glacial', 'not glacial'], range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c'])
	    )
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", ),
	    alt.Y2("Upper CI"),
	    alt.Color(
	        "type:N", 
	        scale=alt.Scale(domain=['fluvial', 'hillslope', 'all', 'glacial', 'not glacial'], range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c'])
	    )
	)

	cum_plot2 = alt.Chart().mark_line(
	    # color='black', 
	    strokeWidth=2.5, 
	    strokeDash=[7, 7]
	    # point={'size':20, 'color': 'black'}
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q', 
	        # scale=alt.Scale(domain=[-5000, 1000])
	    ),
	    alt.Color(
	        "type:N",
	    )
	)

	error_bars2 = alt.Chart().mark_bar(
	    width=2,
	    # color='black'
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title=["Cumulative net volumetric change", "(10³ m³/yr)"]),
	    alt.Y2("Upper CI"),
	    alt.Color(
	        "type:N",
	    )
	)

	bounding_points = alt.Chart().mark_circle(size=100, ).encode(
	    alt.X("end_time:T"),
	    alt.Y("cumulative volume:Q"),
	    alt.Color(
	        "type:N", 
	        scale=alt.Scale(domain=['fluvial', 'hillslope', 'all', 'glacial', 'not glacial'], range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c'])
	    )
	)

	bounding_points2 = alt.Chart().mark_circle(size=100, color='#17becf').encode(
	    alt.X("end_time:T"),
	    alt.Y("cumulative volume:Q")
	)

	cumulative_by_process_chart = alt.layer(
	    cum_plot.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['all', 'fluvial', 'hillslope'])).transform_filter(alt.datum.bounding == False),
	    error_bars.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['all', 'fluvial', 'hillslope'])).transform_filter(alt.datum.bounding == False),
	    cum_plot2.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['not glacial', 'glacial'])).transform_filter(alt.datum.bounding == False),
	    error_bars2.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['not glacial', 'glacial'])).transform_filter(alt.datum.bounding == False),
	    # bounding_points.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['fluvial', 'hillslope', 'all'])).transform_filter(alt.datum.bounding == True),
	    bounding_points2.transform_filter(alt.datum.type == 'glacial').transform_filter(alt.datum.bounding == True),
	    data=src
	).properties(
	    width=350
	).facet(
	    column=alt.Column('valley:N', header=alt.Header(labels=False, labelFontSize=16), title=''),
	)
	cumulative_by_process_chart

	# %%
	if not os.path.exists('outputs/final_figures_data'):
	    os.mkdir('outputs/final_figures_data')
	src.to_csv('outputs/final_figures_data/time_series_cumulative.csv')

	# %%
	src = cum_data_alltogether
	src = src.query("valley != 'Easton'")
	part1 = src[~src['type'].isin(['gully', 'mass wasting'])]
	part2 = src[src['type'].isin(['gully', 'mass wasting'])].groupby(['valley', 'start_time', 'end_time', 'bounding']).sum().reset_index().assign(type='not glacial')
	src = pd.concat([part1, part2])

	src = pd.concat([
	    src,
	    pd.DataFrame({
	        'valley': ['Coleman', 'Deming', 'Mazama', 'Rainbow'],
	        'end_time': [datetime(1947, 9, 14), datetime(1947, 9, 14), datetime(1947, 9, 14), datetime(1947, 9, 14)],
	        'cumulative volume': [0,0,0,0],
	        'bounding': [False, False, False, False],
	        'type': ['not glacial', 'not glacial', 'not glacial', 'not glacial']
	    })
	])

	cum_plot = alt.Chart().mark_line(
	     strokeWidth=2.5,
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q', 
	        scale=alt.Scale(domain=[-4000, 500])
	    ),
	    # alt.Color("type:N")
	    alt.Color(
	        "type:N", 
	        scale=alt.Scale(domain=['fluvial', 'hillslope', 'all', 'glacial', 'not glacial'], range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c'])
	    )
	)

	cum_plot2 = alt.Chart().mark_line(
	    # color='black', 
	    strokeWidth=2.5, 
	    strokeDash=[7, 7]

	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q', 
	        # scale=alt.Scale(domain=[-5000, 1000])
	    ),
	    alt.Color(
	        "type:N",
	    )
	)



	alt.layer(
	    cum_plot.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['all', 'fluvial', 'hillslope'])).transform_filter(alt.datum.bounding == False),
	    
	    cum_plot2.transform_filter(alt.FieldOneOfPredicate(field='type', oneOf=['not glacial', 'glacial'])).transform_filter(alt.datum.bounding == False),
	    
	    data=src
	).properties(
	    width=350
	).facet(
	    column=alt.Column('valley:N', header=alt.Header(labels=False, labelFontSize=16), title=''),
	).configure_legend(
	    titleFontSize=16, 
	    labelFontSize=14, 
	    orient='top', 
	    symbolSize=1000, 
	    symbolStrokeWidth=2
	)

	# %%
	src = cum_data_alltogether
	src = src.query("valley != 'Easton'")
	# src = src.query("valley != 'Coleman'")


	cum_plot = alt.Chart().mark_line(
	        point={'size':20},
	     strokeWidth=2.5,
	).encode(
	    alt.X('end_time:T', title='Time'),
	    alt.Y('cumulative volume:Q', 
	        # scale=alt.Scale(domain=[-4000, 500])
	    ),
	    # alt.Color("type:N")
	    alt.Color("valley:N")
	)

	error_bars = alt.Chart().mark_bar(
	    width=2
	).encode(
	    alt.X("end_time:T"),
	    alt.Y("Lower CI", title="Cumulative net change, in 1,000s of m³/yr"),
	    alt.Y2("Upper CI"),
	    alt.Color("valley:N")
	)


	cumulative_process_facet_valley_color = alt.layer(
	    cum_plot.transform_filter(alt.datum.bounding == False),
	    error_bars.transform_filter(alt.datum.bounding == False),
	    data=src
	).properties(
	    width=350
	).facet(
	    column=alt.Column('type:N', 
	    header=alt.Header(labelFontSize=16), title=''),
	).resolve_scale(y='independent')
	cumulative_process_facet_valley_color

	# %%
	src = gross_data_bytype_df.drop(columns=['index'])
	src = src.query("valley != 'Easton'")
	# src = src.query("valley != 'Coleman'")

	src['Annual Mass Wasted'] = src['Annual Mass Wasted'].apply(lambda x: 1 if x == 0 else x)
	src['Upper CI'] = src['Upper CI'].apply(lambda x: 1 if x == 0 else x)
	src['Lower CI'] = src['Lower CI'].apply(lambda x: 1 if x == 0 else x)

	bars_neg = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	    color='#1f77b4'
	).encode(
	    alt.X('start_time:T', axis=alt.Axis(labels=False), title=None),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	)

	bars_pos = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	    color='#1f77b4',
	    opacity=0.3
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	)

	error_bars = alt.Chart().mark_bar(
	    color="black",
	    width=2
	).encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=["Annual volumetric change", "(10³ m³/yr)"]),
	    alt.Y2("Upper CI")
	)

	fluvial_combo_gross_bars_chart = alt.layer(
	    bars_neg.transform_filter(alt.datum.type == 'negative').transform_filter(alt.datum.process == 'fluvial'), 
	    bars_pos.transform_filter(alt.datum.type == 'positive').transform_filter(alt.datum.process == 'fluvial'), 
	    error_bars.transform_filter(alt.datum.type == 'negative').transform_filter(alt.datum.process == 'fluvial'), 
	    error_bars.transform_filter(alt.datum.type == 'positive').transform_filter(alt.datum.process == 'fluvial'),
	    data=src
	).properties(
	    width=350,
	    height=200
	).facet(
	    column=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labelFontSize=24
	            # labelOrient='top',
	            # labelFontWeight="bold", 
	            # labelPadding=0
	        ),
	        title=None
	    ),
	    # spacing=1
	)
	fluvial_combo_gross_bars_chart 

	# %%
	# scale=alt.Scale(domain=['fluvial', 'hillslope', 'all'], range=['#1f77b4', '#ff7f0e', '#d62728'])

	src = gross_data_bytype_df.drop(columns=['index'])
	src = src.query("valley != 'Easton'")
	# src = src.query("valley != 'Coleman'")

	src['Annual Mass Wasted'] = src['Annual Mass Wasted'].apply(lambda x: 1 if x == 0 else x)
	src['Upper CI'] = src['Upper CI'].apply(lambda x: 1 if x == 0 else x)
	src['Lower CI'] = src['Lower CI'].apply(lambda x: 1 if x == 0 else x)

	bars_neg = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	    color='#d62728'
	).encode(
	    alt.X('start_time:T', axis=alt.Axis(labels=False), title=None),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	)

	bars_pos = alt.Chart().mark_bar(
	    strokeWidth = 2,
	    stroke="white",
	    color='#d62728',
	    opacity=0.3
	).encode(
	    alt.X('start_time:T'),
	    alt.X2('end_time:T'),
	    alt.Y('Annual Mass Wasted'),
	)

	error_bars = alt.Chart().mark_bar(
	    color="black",
	    width=2
	).encode(
	    alt.X('Average Date:T'),
	    alt.Y("Lower CI", title=[
	        # "Annualized rate of volumetric change,", "in 1,000s of m³/yr"
	        ]),
	    alt.Y2("Upper CI")
	)

	hillslope_combo_gross_bars_chart = alt.layer(
	    bars_neg.transform_filter(alt.datum.type == 'negative').transform_filter(alt.datum.process == 'hillslope'), 
	    bars_pos.transform_filter(alt.datum.type == 'positive').transform_filter(alt.datum.process == 'hillslope'), 
	    error_bars.transform_filter(alt.datum.type == 'negative').transform_filter(alt.datum.process == 'hillslope'), 
	    error_bars.transform_filter(alt.datum.type == 'positive').transform_filter(alt.datum.process == 'hillslope'),
	    data=src
	).properties(
	    width=350,
	    height=200
	).facet(
	    column=alt.Column(
	        'valley:N', 
	        header=alt.Header(
	            labels=False,
	            labelFontSize=16
	            # labelOrient='top',
	            # labelFontWeight="bold", 
	            # labelPadding=0
	        ),
	        title=None
	    ),
	    # spacing=1
	)

	hillslope_combo_gross_bars_chart

	# %%
	(
	    fluvial_combo_gross_bars_chart & hillslope_combo_gross_bars_chart & cumulative_by_process_chart
	).configure_legend(
	    titleFontSize=16, 
	    labelFontSize=14, 
	    orient='top', 
	    symbolSize=1000, 
	    symbolStrokeWidth=2
	).configure_axis(
	    labelFontSize=24, 
	    titleFontSize=24
	).resolve_scale(
	    x='shared'
	)

	# %%
	(
	    fluvial_combo_gross_bars_chart & hillslope_combo_gross_bars_chart & cumulative_by_process_chart_no_glacial
	).configure_legend(
	    titleFontSize=16, 
	    labelFontSize=14, 
	    orient='top', 
	    symbolSize=1000, 
	    symbolStrokeWidth=2
	).configure_axis(
	    labelFontSize=24, 
	    titleFontSize=24
	).resolve_scale(
	    x='shared'
	)

	# %%
	src = cum_data_alltogether
	src = src.query("valley != 'Easton'")
	# src = src.query("valley != 'Coleman'")
	src.head()

	# %%
	src.type.unique()


	# %%
	def type_chart(
	    src, 
	    type, 
	    x_axis=alt.Axis(), 
	    facet_column=alt.Column('valley:N')
	):
	    return alt.Chart(src).transform_filter(
	        alt.datum.bounding == False
	    ).transform_filter(
	        alt.datum.type == type
	    ).mark_bar(
	        strokeWidth = 2,
	        stroke="white"
	    ).encode(
	        alt.X('start_time:T', axis=x_axis),
	        alt.X2('end_time:T'),
	        alt.Y('Annual Mass Wasted')
	    ).properties(
	        height=175
	    ).facet(
	        column=facet_column
	    )


	# %%
	alt_col = alt.Column('valley:N', title=None, header=alt.Header(labels=False))
	alt.vconcat(
	    type_chart(src.sample(src.shape[0]), 'hillslope', x_axis=None),
	    type_chart(src.sample(src.shape[0]), 'fluvial', x_axis=None, facet_column=alt_col),
	    type_chart(src.sample(src.shape[0]), 'mass wasting', x_axis=None, facet_column=alt_col),
	    type_chart(src.sample(src.shape[0]), 'gully', x_axis=None, facet_column=alt_col),
	    type_chart(src.sample(src.shape[0]), 'glacial', facet_column=alt_col),
	).resolve_scale(x='shared')




	# %%
	src_bar = cum_data_alltogether.query("bounding == True")
	src_bar = src_bar[src_bar['type'].isin(['glacial', 'gully', 'mass wasting'])]
	src_bar = src_bar.groupby("type").sum(numeric_only=True).reset_index()
	src_bar['cumulative volume'] = - src_bar['cumulative volume']


	# %%
	src_bar

	# %%
	alt.Chart(
	    src_bar
	).mark_bar().encode(
	    alt.X('type:N', axis=alt.Axis(labelAngle=0)),
	    alt.Y("cumulative volume:Q", axis=alt.Axis(tickCount=5), title='Total volumetric change (1,000s m³/yr)')
	).properties(width=600, height=400).configure_legend(
	    titleFontSize=16, 
	    labelFontSize=14, 
	    orient='top', 
	    symbolSize=1000, 
	    symbolStrokeWidth=2
	).configure_axis(
	    labelFontSize=24, 
	    titleFontSize=16
	)

	# %%
	src_bar

	# %%
	src_bar['is_glacial'] = src_bar['type'] == 'glacial'
	alt.Chart(
	    src_bar
	).mark_bar().encode(
	    alt.X('is_glacial:N'),
	    alt.Color("type:N", ),
	    alt.Y("cumulative volume:Q", axis=alt.Axis(tickCount=5))
	).properties(width=600, height=400).configure_legend(
	    titleFontSize=16, 
	    labelFontSize=14, 
	    orient='top', 
	    symbolSize=1000, 
	    symbolStrokeWidth=2
	).configure_axis(
	    labelFontSize=24, 
	    titleFontSize=16
	)

	# %%
	yield_df = pd.read_csv(os.path.join(BASE_PATH, "yield_table.csv"))

	# %%
	yield_df.head(3)

	# %% [markdown]
	# ## Save data

	# %%
	gross_data_bytype_df[[
	    'Annual Mass Wasted',
	    'Upper CI',
	    'Lower CI',
	    'start_time',
	    'end_time',
	    'Average Date',
	    "process",
	    "type",
	    "valley",
	]].to_csv('outputs/final_figures_data/time_series_annualized_gross.csv')

	# %% [markdown]
	# ### Save lithology data for igneous-fraction

	# %%
	lithology_data = lithology_data.query("description != 'Igneous'")
	lithology_data = lithology_data.groupby("Valley Name").sum().reset_index()
	lithology_data

	# %%
	lithology_data.to_csv("outputs/lithology.csv")

	# %%
