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
import xdem
import rioxarray as rix
import xarray as xr
import numpy as np
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt
import altair as alt
from pathlib import Path
from rasterio.enums import Resampling
import pandas as pd
import geopandas as gpd
import geoutils as gu
from pysheds.grid import Grid

import matplotlib.cm
import altair as alt
import copy
import seaborn as sns

if __name__ == "__main__":   

    BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
    print(f"retrieved base path: {BASE_PATH}")

    early_dem_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/1947_09_14.tif")
    late_dem_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dems/2015_09_01.tif")

    dtm_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/reference_dem_highres/baker/2015_dtm_10m.tif")

    initiation_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/erosion_initiation.shp")
    gully_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/gully.shp")
    wasting_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/mass_wasting.shp")
    glacial_debutressing_polygons_fn = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/debutressing.shp")


    glacier_polys_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/**/glaciers.geojson"), recursive=True)
    erosion_polys_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/**/erosion.geojson"), recursive=True)

    dods_output_path = os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/dods/")

    strip_time_format = "%Y_%m_%d"
    reference_dem_date = "2015_09_01"
    reference_dem_date = datetime.strptime(
        reference_dem_date, 
        strip_time_format
    )

    streamstats_watersheds_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/streamstats_watersheds/*.geojson"))

    dod_uncertainty_fn = "outputs/uncertainty_wholemountain.pcl"

    FILTER_OUTLIERS = True
    SIMPLE_FILTER = True
    simple_filter_threshold = 50
    INTERPOLATE = True
    interpolation_max_search_distance = 50
    reasonable_bounds = (580000, 5395000, 595000, 5413000)

    # %% [markdown]
    # # Prepare DDEMs

    # %%
    dem_fn_list = [early_dem_fn, late_dem_fn]
    datetimes = [datetime.strptime(Path(f).stem, strip_time_format) for f in dem_fn_list]

    # %% [markdown]
    # #### Create DEMCollection and calculate DDEMs

    # %%
    demcollection = xdem.DEMCollection.from_files(
        dem_fn_list, 
        datetimes, 
        reference_dem_date, 
        None, 
        10,
        resampling = Resampling.cubic
    )

    # %%
    _ = demcollection.subtract_dems_intervalwise()

    # %% [markdown]
    # #### Remove outliers and then interpolate

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

    if INTERPOLATE:
        interpolated_ddems = demcollection.interpolate_ddems(max_search_distance=interpolation_max_search_distance)
        demcollection.set_ddem_filled_data()

    # %% [markdown]
    # #### Open glacier polygons and remove glacier signals

    # %%
    from pprint import pprint

    # %%
    all_glaciers_gdf = pd.concat([gpd.read_file(f) for f in glacier_polys_fns]).to_crs(demcollection.dems[0].crs)
    formatted_timestamps = [datetime.strftime(pd.to_datetime(ts), strip_time_format) for ts in demcollection.timestamps]
    glaciers_gdf = all_glaciers_gdf[all_glaciers_gdf.year.isin(formatted_timestamps)]
    glaciers_gdf['date'] = glaciers_gdf['year'].apply(lambda x: datetime.strptime(x, strip_time_format))

    # %%
    for ddem in demcollection.ddems:
        ddem
        relevant_glaciers_gdf = glaciers_gdf[glaciers_gdf['date'].isin([ddem.interval.left, ddem.interval.right])]
        relevant_glaciers_mask = gu.Vector(relevant_glaciers_gdf).create_mask(ddem).squeeze()
        ddem.data.mask = np.logical_or(ddem.data.mask, relevant_glaciers_mask)

    # %%
    import math


    # %%
    demcollection.plot_ddems(max_cols=2, figsize=(20,10))
    plt.gca().set_aspect('equal')

    # %% [markdown]
    # # Prepare terrain attributes

    # %% [markdown]
    # #### Use xdem for some

    # %%
    dtm = xdem.DEM(dtm_fn)


    # %%
    def plot_attribute(attribute, cmap = 'Blues', label=None, vlim=None):

        add_cb = True if label is not None else False

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        if vlim is not None:
            if isinstance(vlim, (int, float)):
                vlims = {"vmin": -vlim, "vmax": vlim}
            elif len(vlim) == 2:
                vlims = {"vmin": vlim[0], "vmax": vlim[1]}
        else:
            vlims = {}

        # attribute.show(
        #     ax=ax,
        #     cmap=cmap,
        #     add_cb=add_cb,
        #     cb_title=label,
        #     **vlims,
        # )

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.show(block=False)


    # %%
    slope = xdem.terrain.slope(dtm)
    aspect = xdem.terrain.aspect(dtm)
    curvature = xdem.terrain.curvature(dtm)

    # %%
    plot_attribute(slope, label='slope'.title())
    plot_attribute(aspect, label='aspect'.title())
    plot_attribute(curvature, label='curvature'.title(), vlim=1)

    # %% [markdown]
    # #### Use pysheds for drainage area

    # %%
    # grid_pysheds = Grid.from_raster(dtm_fn, 'dem')
    # dem_pysheds = grid_pysheds.read_raster(dtm_fn, 'dem')

    # pit_filled_dem_pysheds = grid_pysheds.fill_pits(dem_pysheds)
    # flooded_dem_pysheds = grid_pysheds.fill_depressions(pit_filled_dem_pysheds)
    # inflated_dem_pysheds = grid_pysheds.resolve_flats(flooded_dem_pysheds)
    # fdir_pysheds = grid_pysheds.flowdir(inflated_dem_pysheds) #May need to pass in a dirmap dirmap=dirmap
    # acc_pysheds = grid_pysheds.accumulation(fdir_pysheds)

    # %%
    # import tempfile
    # with tempfile.NamedTemporaryFile() as tmp:
    #     Grid.from_raster(acc_pysheds).to_raster(acc_pysheds, tmp.name)
    #     darea = rix.open_rasterio(tmp.name)

    # %% [markdown]
    # ### Adjust drainage area by pixel area

    # %%
    # darea = darea*np.abs(dtm.res[0]*dtm.res[1])

    # %%
    # darea.rio.to_raster('drainage_area.tif')

    # %%
    darea = rix.open_rasterio(
        os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/whole_mountain/drainage_area_2015.tif")
    )

    # %% [markdown]
    # # Create Dataset

    # %%
    # ddem = demcollection.ddems[0].to_xarray(masked=True)
    # ddem.data = demcollection.ddems[0].data.filled(np.nan)

    # dtm = rix.open_rasterio(dtm_fn, masked=True)

    ddem = demcollection.ddems[0].to_xarray()
    ddem.data = demcollection.ddems[0].data.filled(np.nan)

    dtm = rix.open_rasterio(dtm_fn)

    # %%
    for k,v in {
        'dtm': dtm,
        'ddem': ddem,
        'drainage area': darea,
        'slope': slope.to_xarray(),
        'aspect': aspect.to_xarray(),
        'curvature': curvature.to_xarray()
    }.items():
        print(k)
        print(v.shape)
        print(v.rio.crs)
        print(v.rio.resolution())
        print()

    # %%
    dataset = xr.Dataset({
        'dtm': dtm,
        'ddem': ddem.rio.reproject_match(dtm),
        'drainage area': darea,
        'slope': slope.to_xarray(),
        'aspect': aspect.to_xarray(),
        'curvature': curvature.to_xarray()
    })

    # %% [markdown]
    # # Clip dataset to bounds and remove glacier signals

    # %%
    dataset = dataset.rio.clip_box(*reasonable_bounds)
    dataset = dataset.rio.clip(glaciers_gdf.geometry, invert=True)

    # %% [markdown]
    # # Plot Rasters

    # %%
    fig, axes = plt.subplots(2, len(dataset.data_vars), figsize=(6*len(dataset.data_vars), 6*2))

    cmap = copy.copy(matplotlib.cm.RdYlBu)
    _ = cmap.set_bad('grey')
    from matplotlib import colors

    for i in range(0, len(dataset.data_vars)):
        if list(dataset.data_vars)[i] == 'drainage area':
            axes[0, i].imshow(dataset['drainage area'].values.squeeze(), norm=colors.LogNorm())
        else:
            dataset[list(
                dataset.data_vars)[i]
            ].plot(ax = axes[0, i], cmap=cmap)
        axes[0, i].set_title(list(dataset.data_vars)[i])
        sns.distplot(dataset[list(dataset.data_vars)[i]].values, ax = axes[1, i])
        # histplot instead of distplot
        # and add the keyword args  kde=True, stat="density", linewidth=0
        # sns.histplot(dataset[list(dataset.data_vars)[i]].values.squeeze(), ax = axes[1, i], kde=True, stat="density", linewidth=0)

    plt.tight_layout()
    plt.show(block=False)

    # %% [markdown]
    # # Open erosion polygons

    # %%
    erosion_gdf = pd.concat([gpd.read_file(f) for f in erosion_polys_fns])

    # %%
    erosion_gdf.plot()

    # %%
    dataset

    # %% [markdown]
    # # Slope-Area Analysis

    # %% [markdown]
    # ## Terrain Characteristics of Hillslope and Fluvial areas

    # %%
    import numpy as np

    # %%
    darea_bins = [10**x for x in np.arange(2, 10, 0.5)]

    # %%
    darea_bins


    # %%
    def get_df(ds):
        return ds.to_dataframe().reset_index().drop(columns=['band', 'spatial_ref'])
    def binned_medians(df):
        binned_medians_df = pd.DataFrame(df.groupby(pd.cut(df['drainage area'], bins=darea_bins))['slope'].median()).reset_index()
        binned_medians_df['drainage area'] = binned_medians_df['drainage area'].apply(lambda interval: np.mean([interval.left, interval.right]))
        return binned_medians_df

    all_data_df = get_df(dataset)
    hillslope_data_df = get_df(dataset.rio.clip(erosion_gdf.query("type == 'hillslope'").geometry))
    fluvial_data_df = get_df(dataset.rio.clip(erosion_gdf.query("type == 'fluvial'").geometry))

    all_data_binned_df = binned_medians(all_data_df)
    hillslope_data_binned_df = binned_medians(hillslope_data_df)
    fluvial_data_binned_df = binned_medians(fluvial_data_df)

    # %%
    fig, axes = plt.subplots(1,3, figsize=(15,5), sharex=True, sharey=True)
    axes[0].scatter(all_data_df['drainage area'], all_data_df['slope'], s=0.00001, color='k')
    axes[1].scatter(hillslope_data_df['drainage area'], hillslope_data_df['slope'], s=0.001, color='red', label='hillslope')
    axes[1].scatter(fluvial_data_df['drainage area'], fluvial_data_df['slope'], s=0.01, color='blue', label='fluvial')

    axes[2].scatter(all_data_binned_df['drainage area'], all_data_binned_df['slope'], color='k', edgecolors= "white", label='all pixels')
    axes[2].scatter(hillslope_data_binned_df['drainage area'], hillslope_data_binned_df['slope'], color='red', edgecolors= "white", label='hillslope')
    axes[2].scatter(fluvial_data_binned_df['drainage area'], fluvial_data_binned_df['slope'], color='blue', edgecolors= "white", label='fluvial')

    for ax in axes:
        ax.set_xlim(10,10e6)
        # ax.set_ylim(0,1.5)
        ax.set_xscale('log')
        ax.set_xlabel("Drainage Area (m^2)")
        ax.set_ylabel("Slope (degrees)")

    axes[0].set_title("All pixels")
    axes[1].set_title("Pixels inside erosion polygons")
    lgnd1 = axes[1].legend(loc="upper right")
    axes[2].set_title("Median slope of binned Drainage Area values")
    lgnd2 = axes[2].legend(loc="upper right")

    lgnd1.legendHandles[0]._sizes = [30]
    lgnd1.legendHandles[1]._sizes = [30]
    lgnd2.legendHandles[0]._sizes = [30]
    lgnd2.legendHandles[1]._sizes = [30]
    lgnd2.legendHandles[2]._sizes = [30]

    plt.show(block=False)

    # %%
    fig, axes = plt.subplots(1,2, figsize=(20,10), sharex=True, sharey=True)
    axes[0].scatter(all_data_df['drainage area'], all_data_df['slope'], s=0.00001, color='k')
    axes[1].scatter(hillslope_data_df['drainage area'], hillslope_data_df['slope'], s=0.001, color='red', label='hillslope')
    axes[1].scatter(fluvial_data_df['drainage area'], fluvial_data_df['slope'], s=0.01, color='blue', label='fluvial')

    for ax in axes:
        ax.set_xlim(10,10e6)
        # ax.set_ylim(0,1.5)
        ax.set_xscale('log')
        ax.set_xlabel("Drainage Area (m^2)")
        ax.set_ylabel("Slope (degrees)")

    axes[0].set_title("All pixels")
    axes[1].set_title("Pixels inside erosion polygons")
    lgnd1 = axes[1].legend(loc="upper right")

    lgnd1.legendHandles[0]._sizes = [30]
    lgnd1.legendHandles[1]._sizes = [30]

    plt.show(block=False)

    # %% [markdown]
    # ## Terrain Characteristics of Erosion Initiation Sites

    # %%
    initiation_gdf = gpd.read_file(initiation_polygons_fn)
    initiation_gdf = initiation_gdf.to_crs(dataset.rio.crs)

    # %%
    mean_results = initiation_gdf.apply(
        (lambda row: dataset.rio.clip([row['geometry']]).to_dataframe().mean()),
        axis='columns', result_type='expand'
    )
    initiation_gdf[mean_results.columns] = mean_results

    # %%
    initiation_gdf['type'].unique()

    # %%
    plt.scatter(all_data_binned_df['drainage area'], all_data_binned_df['slope'], color='k', edgecolors= "white")
    plt.scatter(hillslope_data_binned_df['drainage area'], hillslope_data_binned_df['slope'], color='red', edgecolors= "white")
    plt.scatter(fluvial_data_binned_df['drainage area'], fluvial_data_binned_df['slope'], color='blue', edgecolors= "white")
    plt.scatter(initiation_gdf['drainage area'], initiation_gdf['slope'], color='purple', edgecolors= "white", marker='+')
    plt.xscale('log')
    plt.show(block=False)

    # %%
    plt.scatter(all_data_df['drainage area'], all_data_df['slope'], s=0.00001, color='k')

    # %%
    plt.plot(
        all_data_binned_df['drainage area'], 
        all_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
        label = 'All pixels'
    )
    plt.plot(
        hillslope_data_binned_df['drainage area'], 
        hillslope_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
        label = 'Hillslope erosion pixels'
    )
    plt.plot(
        fluvial_data_binned_df['drainage area'], 
        fluvial_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
        label = 'Fluvial erosion pixels'
    )

    plt.scatter(
        initiation_gdf.query("type == 'gully'")['drainage area'], 
        initiation_gdf.query("type == 'gully'")['slope'], 
        color='black', 
        edgecolors= "white", 
        marker='1',
        label = "Gully erosion"
    )
    plt.scatter(
        initiation_gdf.query("type == 'channel'")['drainage area'], 
        initiation_gdf.query("type == 'channel'")['slope'], 
        color='blue', 
        edgecolors= "white", 
        marker='v',
        label = "Channel erosion"
    )
    plt.scatter(
        initiation_gdf.query("type == 'rockfall'")['drainage area'], 
        initiation_gdf.query("type == 'rockfall'")['slope'], 
        color='red', 
        edgecolors= "white", 
        marker='+',
        label = "Rockfall erosion"
    )
    plt.xscale('log')
    plt.xlabel("Drainage Area")
    plt.ylabel("Slope")
    plt.xlim(0.1, 10**6)
    plt.title("Slope-Area Diagram for 1970-2015 Elevation Change ")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show(block=False)

    # %% [markdown]
    # ## Terrain Characteristics of Process Polygons (Gully, Mass Wasting, Fluvial, Glacial  and Mass Wasting sies

    # %%
    gully_gdf = gpd.read_file(gully_polygons_fn)
    gully_gdf['process'] = 'gully'
    wasting_gdf = gpd.read_file(wasting_polygons_fn)
    wasting_gdf['process'] = 'mass wasting'
    glacial_debutress_gdf = gpd.read_file(glacial_debutressing_polygons_fn)
    glacial_debutress_gdf['process'] = 'glacial'

    fluvial_gdf = erosion_gdf.query("type == 'fluvial'")[['name', 'type', 'geometry']]
    fluvial_gdf['process'] = 'fluvial'

    process_gdf = pd.concat([gully_gdf, wasting_gdf, glacial_debutress_gdf, fluvial_gdf]).reset_index(drop=True)

    # %%
    mean_results = process_gdf.apply(
        (lambda row: dataset.rio.clip([row['geometry']]).to_dataframe().agg({
            'ddem': 'mean',
            'dtm': 'mean',
            'drainage area': 'max',
            'slope': 'mean',
            'aspect': 'mean',
            'curvature': 'mean',
        })),
        axis='columns', result_type='expand'
    )
    process_gdf[mean_results.columns] = mean_results
    process_gdf['area'] = process_gdf.geometry.area
    process_gdf['ddem normalized'] = process_gdf['ddem']/process_gdf['area']

    # %%
    process_gdf

    # %%
    plt.figure(figsize=(8, 6), dpi=80)
    plt.plot(
        all_data_binned_df['drainage area'], 
        all_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
        label = 'All pixels'
    )
    plt.plot(
        hillslope_data_binned_df['drainage area'], 
        hillslope_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
        label = 'Hillslope erosion pixels'
    )
    plt.plot(
        fluvial_data_binned_df['drainage area'], 
        fluvial_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
        label = 'Fluvial erosion pixels'
    )

    plt.scatter(
        process_gdf.query("process == 'gully'")['drainage area'], 
        process_gdf.query("process == 'gully'")['slope'], 
        color='purple', 
        edgecolors= "white", 
        marker='1',
        label = "Gully erosion"
    )
    plt.scatter(
        process_gdf.query("process == 'mass wasting'")['drainage area'], 
        process_gdf.query("process == 'mass wasting'")['slope'], 
        color='blue', 
        edgecolors= "white", 
        marker='v',
        label = "Mass wasting"
    )
    plt.xscale('log')
    plt.xlabel("Drainage Area")
    plt.ylabel("Slope")
    plt.title("Slope-Area Diagram for 1970-2015 Elevation Change ")
    plt.legend()
    plt.show(block=False)

    # %%
    src = process_gdf[process_gdf.process.isin(["gully", "mass wasting"])]
    darea_plot = alt.Chart(src).mark_point().encode(
        alt.X('drainage area:Q', scale=alt.Scale(type='log')),
        alt.Y('ddem:Q'),
        alt.Color("process:N")
    )
    slope_plot = alt.Chart(src).mark_point().encode(
        alt.X('slope:Q', scale=alt.Scale(zero=False)),
        alt.Y('ddem:Q'),
        alt.Color("process:N")
    )
    curv_plot = alt.Chart(src).mark_point().encode(
        alt.X('curvature:Q', scale=alt.Scale(zero=False)),
        alt.Y('ddem:Q'),
        alt.Color("process:N")
    )
    (darea_plot) | (slope_plot) | curv_plot

    # %%
    all_data_binned_df['type'] = 'All pixels'
    hillslope_data_binned_df['type'] = 'Hillslope area'
    fluvial_data_binned_df['type'] = 'Fluvial area'

    binned_data = pd.concat([
        all_data_binned_df,
        hillslope_data_binned_df,
        fluvial_data_binned_df,
    ])

    lines = alt.Chart(binned_data).mark_line().encode(
        alt.X("drainage area:Q", scale=alt.Scale(type='log'), title='Drainage area (square meters)'),
        alt.Y("slope:Q"),
        alt.Color(
            "type:N", 
            scale=alt.Scale(
                domain=['Fluvial area', 'Hillslope area', 'All pixels', 'glacial', 'not glacial'], 
                range=['#1f77b4', '#d62728', '#000000', '#17becf', '#2ca02c']
            )
        )
    )

    src = process_gdf[process_gdf.process.isin(["gully", "mass wasting", "glacial"])]
    src.process = src.process.apply(lambda x: 'not glacial' if x in ['gully', 'mass wasting'] else 'glacial')

    points = alt.Chart(src).mark_point().encode(
        alt.X("drainage area:Q", scale=alt.Scale(type='log', domain=[100, 10000000], clamp=True)),
        alt.Y("slope:Q", title = 'Slope (degrees)'),
        alt.Color("process:N"),
        shape = alt.Shape("process:N", scale=alt.Scale(range=['circle', 'triangle-right']))
    )

    lines_and_points_plot = (lines + points).configure_axis(grid=True)
    lines_and_points_plot

    # %%
    fig, axes = plt.subplots(1,2, figsize=(20,10), sharex=True, sharey=True)
    axes[0].scatter(all_data_df['drainage area'], all_data_df['slope'], s=0.00001, color='k')
    axes[1].scatter(hillslope_data_df['drainage area'], hillslope_data_df['slope'], s=0.001, color='red', label='hillslope')
    axes[1].scatter(fluvial_data_df['drainage area'], fluvial_data_df['slope'], s=0.01, color='blue', label='fluvial')

    for ax in axes:
        ax.set_xlim(10,10e6)
        # ax.set_ylim(0,60)
        ax.set_xscale('log')
        ax.set_xlabel("Drainage Area (m^2)")
        ax.set_ylabel("Slope (degrees)")

    axes[0].set_title("All pixels")
    axes[1].set_title("Pixels inside erosion polygons")
    lgnd1 = axes[1].legend(loc="upper right")

    lgnd1.legendHandles[0]._sizes = [30]
    lgnd1.legendHandles[1]._sizes = [30]

    plt.show(block=False)

    # %%
    src.process.unique()

    # %%
    fig, axes = plt.subplots(1,2, figsize=(20,10), sharex=True, sharey=True)
    axes[0].scatter(all_data_df['drainage area'], all_data_df['slope'], s=0.00001, color='k')
    axes[1].scatter(hillslope_data_df['drainage area'], hillslope_data_df['slope'], s=0.001, color='red', label='hillslope')
    axes[1].scatter(fluvial_data_df['drainage area'], fluvial_data_df['slope'], s=0.01, color='blue', label='fluvial')

    axes[1].scatter(
        src.query('process == "not glacial"')['drainage area'], src.query('process == "not glacial"')['slope'], color='green', marker='>', label='hillslope'
    )
    axes[1].scatter(src.query('process == "glacial"')['drainage area'], src.query('process == "glacial"')['slope'], color='purple', marker='P', label='hillslope')

    axes[1].plot(
        hillslope_data_binned_df['drainage area'], 
        hillslope_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
    )
    axes[1].plot(
        fluvial_data_binned_df['drainage area'], 
        fluvial_data_binned_df['slope'], 
        color='k', 
        # s=20, 
        # edgecolors= "white",
    )

    for ax in axes:
        ax.set_xlim(10,10e6)
        ax.set_ylim(0,60)
        ax.set_xscale('log')
        ax.set_xlabel("Drainage Area (m^2)")
        ax.set_ylabel("Slope (degrees)")

    axes[0].set_title("All pixels")
    axes[1].set_title("Pixels inside erosion polygons")
    lgnd1 = axes[1].legend(loc="upper right")

    lgnd1.legendHandles[0]._sizes = [30]
    lgnd1.legendHandles[1]._sizes = [30]

    plt.show(block=False)

    # %%

    fig, ax = plt.subplots()
    ax.scatter(hillslope_data_df['drainage area'], hillslope_data_df['slope'], s=0.025,  color='orange', label='Hillslope domain')
    ax.scatter(fluvial_data_df['drainage area'], fluvial_data_df['slope'], s=0.05, color='#1f77b4', label='Fluvial domain')

    points_src = src[src['process'].isin(['glacial', 'not glacial'])]
    ax.scatter(points_src['drainage area'], points_src['slope'], facecolor='none', edgecolor='black', linewidth=2, marker='o', s=50, label='Erosion polygons')

    ax.plot(
        hillslope_data_binned_df['drainage area'], 
        hillslope_data_binned_df['slope'], 
        color='orange', 
        # alpha=0.5,
        # s=20, 
        # edgecolors= "white",
    )
    ax.plot(
        fluvial_data_binned_df['drainage area'], 
        fluvial_data_binned_df['slope'], 
        color='#1f77b4', 
        # alpha=0.5,
        # s=20, 
        # edgecolors= "white",
    )


    ax.set_xlim(100,10e6)
    ax.set_ylim(0,60)
    ax.set_xscale('log')
    ax.set_xlabel("Drainage Area (m²)", fontsize=12)
    ax.set_ylabel("Slope (degrees)", fontsize=12)
    lgnd1 = ax.legend(loc="upper right")

    lgnd1.legendHandles[0]._sizes = [30]
    lgnd1.legendHandles[1]._sizes = [30]
    # plt.grid(True, which='both', alpha=0.5)
    plt.gca().set_axisbelow(True)
    plt.show(block=False)

    # %%
    src = pd.concat([
        hillslope_data_df,
        fluvial_data_df
    ]).dropna()

    # %%
    len(src.dropna()[src['slope'] > 34]) / len(src.dropna()['slope'])

    # %%
    len(src.dropna()[src['slope'] > 19]) / len(src.dropna()['slope'])

    # %%
    len(hillslope_data_df.dropna()[hillslope_data_df['slope'] > 34]) / len(hillslope_data_df.dropna()['slope'])

    # %%
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(4.8, 4.8))
    sns.distplot(hillslope_data_df['slope'], ax=axes[0], label='Hillslope Domain', color='orange')
    axes[0].annotate("Hillslope Domain", xy=(79,0.03), horizontalalignment='right', fontsize=12)
    sns.distplot(fluvial_data_df['slope'], ax=axes[1], label='Fluvial Domain')
    axes[1].annotate("Fluvial Domain", xy=(79,0.05), horizontalalignment='right', fontsize=12)
    plt.xlabel("Slope", fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.xlim(0,80)

    # %%
    len(points_src.query("slope > 34")), len(points_src.query("slope <= 34"))

    # %%
    lia_outlet_intersections_gdf = gpd.read_file(
        os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/lia_outlet_intersections.shp")
    )

    # %%
    lia_outlet_intersections_gdf

    # %% [markdown]
    # for i, row in lia_outlet_intersections_gdf.iterrows():
    #     print(row['valley'])
    #     print(dataset['drainage area'].rio.clip([row['geometry']]).max())
    #     print(dataset['slope'].rio.clip([row['geometry']]).max())

    # %%
    src

    # %% [markdown]
    # ## Terrain characteristics of watersheds, considering all area within watershed

    # %% [markdown]
    # ### Prep some DDEM raster datasets

    # %% [markdown]
    # #### Read in uncertainty parameters for thresholding

    # %%
    uncertainty_df = pd.read_pickle(dod_uncertainty_fn)
    uncertainty_df = uncertainty_df[uncertainty_df['Start Date'] == demcollection.ddems[0].start_time]
    uncertainty_df = uncertainty_df[uncertainty_df['End Date'] == demcollection.ddems[0].end_time]
    assert len(uncertainty_df) == 1
    low = uncertainty_df['90% CI Lower Bound'].iloc[0]
    hi = uncertainty_df['90% CI Upper Bound'].iloc[0]

    # %% [markdown]
    # #### Create new datasets: 
    #
    # * ddem erosion (elevation changes in erosion areas only)
    # * ddem hillslope (elevation changes in hillslope erosion areas only)
    # * ddem fluvial (elevation changes in fluvial erosion areas only)
    # * ddem negative (negative elevation changes in all areas)
    # * ddem erosion negative (negative elevation changes in erosion areas only all areas)
    # * ddem hillslope negative (negative elevation changes in hillslope erosion areas only all areas)
    # * ddem fluvial negative (negative elevation changes in fluvial erosion areas only all areas)

    # %%
    dataset

    # %%
    dataset['ddem erosion'] = dataset['ddem'].rio.clip(erosion_gdf.geometry)
    # dataset['ddem erosion'] = dataset['ddem erosion'].where(dataset['ddem erosion'] != dataset['ddem erosion'].attrs['_FillValue'])
    dataset['ddem hillslope'] = dataset['ddem'].rio.clip(erosion_gdf.query("type == 'hillslope'").geometry)
    # dataset['ddem hillslope'] = dataset['ddem hillslope'].where(dataset['ddem hillslope'] != dataset['ddem hillslope'].attrs['_FillValue'])
    dataset['ddem fluvial'] = dataset['ddem'].rio.clip(erosion_gdf.query("type == 'fluvial'").geometry)
    # dataset['ddem fluvial'] = dataset['ddem fluvial'].where(dataset['ddem fluvial'] != dataset['ddem fluvial'].attrs['_FillValue'])

    dataset['ddem negative'] = xr.where(dataset['ddem'] < 0, dataset['ddem'], np.nan)
    dataset['ddem negative'] = xr.where(
        np.logical_or(dataset['ddem negative'] < low, np.isnan(dataset['ddem negative'])), 
        dataset['ddem negative'], 
        0
    )

    dataset['ddem erosion negative'] = xr.where(dataset['ddem erosion'] < 0, dataset['ddem erosion'], np.nan)
    dataset['ddem erosion negative'] = xr.where(
        np.logical_or(dataset['ddem erosion negative'] < low, np.isnan(dataset['ddem erosion negative'])), 
        dataset['ddem erosion negative'], 
        0
    )

    dataset['ddem hillslope negative'] = xr.where(dataset['ddem hillslope'] < 0, dataset['ddem hillslope'], np.nan)
    dataset['ddem hillslope negative'] = xr.where(
        np.logical_or(dataset['ddem hillslope negative'] < low, np.isnan(dataset['ddem hillslope negative'])), 
        dataset['ddem hillslope negative'], 
        0
    )

    dataset['ddem fluvial negative'] = xr.where(dataset['ddem fluvial'] < 0, dataset['ddem fluvial'], np.nan)
    dataset['ddem fluvial negative'] = xr.where(
        np.logical_or(dataset['ddem fluvial negative'] < low, np.isnan(dataset['ddem fluvial negative'])), 
        dataset['ddem fluvial negative'], 
        0
    )

    # %% [markdown]
    # #### Plot new dataset variables

    # %%
    fig, axes = plt.subplots(
        2, 
        len(dataset.data_vars)-6, 
        figsize=(6*(len(dataset.data_vars)-6), 6*2)
    )

    cmap = copy.copy(matplotlib.cm.RdYlBu)
    _ = cmap.set_bad('grey')
    from matplotlib import colors

    for i, var in enumerate(list(dataset.data_vars)[6:]):
        if var == "drainage area":
            axes[0, i].imshow(dataset[var].values.squeeze(), norm=colors.LogNorm())
        else:
            dataset[var].plot(ax = axes[0, i], cmap=cmap)
        axes[0, i].set_title(str(var))
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
        sns.distplot(dataset[var].values, ax = axes[1, i])
    plt.tight_layout()
    plt.show(block=False)

    # %% [markdown]
    # ### Open StreamStats watersheds
    #

    # %%
    gdf = gpd.GeoDataFrame()
    for f in streamstats_watersheds_fns:
        new_data = gpd.read_file(f)
        new_data['Valley Name'] = f.split("/")[-1].split(".geojson")[0]
        gdf = gpd.GeoDataFrame(pd.concat([gdf, new_data]))

    gdf = gdf.to_crs(dataset['ddem'].rio.crs)
    nhd_df = gdf[gdf.geometry.type != "Point"]

    # %%
    nhd_df.plot()

    # %%
    for var in dataset.data_vars:
        values = dataset[var].values.copy()
        values[
            np.isnan(dataset.data_vars['ddem'].values)
        ] = np.nan
        dataset[var].values = values

    # %% [markdown]
    # ### Calculate watershed statistics with watershed polygons

    # %%
    mean_results = nhd_df.apply(
        (lambda row: dataset.rio.clip([row['geometry']]).to_dataframe().agg({
                'dtm': 'mean',
                'drainage area': 'max',
                'slope': 'mean',
                'aspect': 'mean',
                'curvature': 'mean',
                
                'ddem': 'sum',
                'ddem erosion': 'sum',
                'ddem hillslope': 'sum',
                'ddem fluvial': 'sum',
                'ddem negative': 'sum',
                'ddem erosion negative': 'sum',
                'ddem hillslope negative': 'sum',
                'ddem fluvial negative': 'sum',
                
            }
        )),
        axis='columns', result_type='expand'
    )
    nhd_df[mean_results.columns] = mean_results

    # %% [markdown]
    # ### Calculate glacial area in start year, end year, max year (1979) and changes in that area

    # %%
    glaciers_start = all_glaciers_gdf[all_glaciers_gdf.year == datetime.strftime(pd.to_datetime(demcollection.timestamps[0]), strip_time_format)]
    glaciers_end = all_glaciers_gdf[all_glaciers_gdf.year == datetime.strftime(pd.to_datetime(demcollection.timestamps[-1]), strip_time_format)]
    glaciers_max = gpd.GeoDataFrame(pd.concat([
        all_glaciers_gdf.query("year == '1979_10_06'"),
        all_glaciers_gdf.query("year == '1977_09_27' and Name == 'Rainbow Glacier WA'")
    ]))

    nhd_df['starting glacial area'] = nhd_df.geometry.apply(lambda watershed_geom: glaciers_start.geometry.apply(lambda geo: watershed_geom.intersection(geo)).area.sum())
    nhd_df['ending glacial area'] = nhd_df.geometry.apply(lambda watershed_geom: glaciers_end.geometry.apply(lambda geo: watershed_geom.intersection(geo)).area.sum())
    nhd_df['max glacial area'] = nhd_df.geometry.apply(lambda watershed_geom: glaciers_max.geometry.apply(lambda geo: watershed_geom.intersection(geo)).area.sum())
    nhd_df['glacial advance area'] = nhd_df['max glacial area'] - nhd_df['starting glacial area']
    nhd_df['glacial retreat area'] = nhd_df['ending glacial area'] - nhd_df['max glacial area']

    # %%
    nhd_df[[
        'Valley Name',
        'starting glacial area',
        'max glacial area',
        'ending glacial area',
        'glacial advance area',
        'glacial retreat area'  
    ]]

    # %%
    pixel_area = np.abs(dataset.rio.resolution()[0]*dataset.rio.resolution()[1])

    # %%
    nhd_df['ddem'] = nhd_df['ddem']*pixel_area
    nhd_df['ddem erosion'] = nhd_df['ddem erosion']*pixel_area
    nhd_df['ddem hillslope'] = nhd_df['ddem hillslope']*pixel_area
    nhd_df['ddem fluvial'] = nhd_df['ddem fluvial']*pixel_area
    nhd_df['ddem negative'] = nhd_df['ddem negative']*pixel_area
    nhd_df['ddem erosion negative'] = nhd_df['ddem erosion negative']*pixel_area
    nhd_df['ddem hillslope negative'] = nhd_df['ddem hillslope negative']*pixel_area
    nhd_df['ddem fluvial negative'] = nhd_df['ddem fluvial negative']*pixel_area

    # %% [markdown]
    # ### Calculate erosion measurement area and incision rates

    # %%
    # np.count_nonzero(~np.isnan(data))
    nhd_df['area'] = nhd_df.apply(
        lambda row: np.count_nonzero(~np.isnan(dataset['ddem'].rio.clip([row['geometry']]).values))*pixel_area, axis='columns'
    )
    nhd_df['area erosion'] = nhd_df.apply(
        lambda row: np.count_nonzero(~np.isnan(dataset['ddem erosion'].rio.clip([row['geometry']]).values))*pixel_area, axis='columns'
    )
    nhd_df['area hillslope'] = nhd_df.apply(
        lambda row: np.count_nonzero(~np.isnan(dataset['ddem hillslope'].rio.clip([row['geometry']]).values))*pixel_area, axis='columns'
    )
    nhd_df['area fluvial'] = nhd_df.apply(
        lambda row: np.count_nonzero(~np.isnan(dataset['ddem fluvial'].rio.clip([row['geometry']]).values))*pixel_area, axis='columns'
    )

    # %%
    nhd_df['ddem incision'] = nhd_df['ddem'] / nhd_df['area']
    nhd_df['ddem erosion incision'] = nhd_df['ddem erosion'] / nhd_df['area erosion']
    nhd_df['ddem hillslope incision'] = nhd_df['ddem hillslope'] / nhd_df['area hillslope']
    nhd_df['ddem fluvial incision'] = nhd_df['ddem fluvial'] / nhd_df['area fluvial']

    nhd_df['ddem negative incision'] = nhd_df['ddem negative'] / nhd_df['area']
    nhd_df['ddem erosion negative incision'] = nhd_df['ddem erosion negative'] / nhd_df['area erosion']
    nhd_df['ddem hillslope negative incision'] = nhd_df['ddem hillslope negative'] / nhd_df['area hillslope']
    nhd_df['ddem fluvial negative incision'] = nhd_df['ddem fluvial negative'] / nhd_df['area fluvial']

    # %%
    nhd_df

    # %% [markdown]
    # ### Plot: pair plots

    # %%
    src = nhd_df[[
        'Valley Name',
        'dtm',
        'drainage area',
        'slope',
        'aspect',
        'curvature',
        'max glacial area',
        'glacial advance area',
        'glacial retreat area',
        'ddem',
        'ddem erosion',
        'ddem hillslope',
        'ddem fluvial',
    ]]


    g = sns.PairGrid(src, hue ='Valley Name')
    # g = g.map_diag(plt.hist, histtype="step", linewidth=3)
    g = g.map_offdiag(plt.scatter)
    g = g.add_legend()

    # %%
    src = nhd_df[[
        'Valley Name',
        'dtm',
        'drainage area',
        'slope',
        'aspect',
        'curvature',
        'max glacial area',
        'glacial advance area',
        'glacial retreat area',
        'ddem incision',
        'ddem erosion incision',
        'ddem hillslope incision',
        'ddem fluvial incision',
    ]]

    src = src[src["Valley Name"] != 'thunder']


    g = sns.PairGrid(src, hue ='Valley Name')
    # g = g.map_diag(plt.hist, histtype="step", linewidth=3)

    g = g.map_offdiag(plt.scatter)
    g = g.add_legend()

    # %%
    src = nhd_df[[
        'Valley Name',
        'dtm',
        'drainage area',
        'slope',
        'aspect',
        'curvature',
        'max glacial area',
        'glacial advance area',
        'glacial retreat area',
        'ddem negative',
        'ddem erosion negative',
        'ddem hillslope negative',
        'ddem fluvial negative',
    ]]


    g = sns.PairGrid(src, hue ='Valley Name')
    # g = g.map_diag(plt.hist, histtype="step", linewidth=3)
    g = g.map_offdiag(plt.scatter)
    g = g.add_legend()

    # %%
    src

    # %%
    chart = alt.Chart(src).mark_circle(size=100).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False), title='Drainage Area at Watershed Outlet'),
        alt.Y('slope:Q', scale=alt.Scale(zero=False), title='Watershed Mean Slope'),
        alt.Color('ddem erosion negative incision:Q', scale=alt.Scale(scheme='viridis'))
    )
    annotations1 = alt.Chart(src).mark_text(
        align='right',
        baseline='middle',
        fontSize = 14,
        dx = -7
    ).transform_filter(
        alt.FieldRangePredicate(field='drainage area', range=[100000, 10000000])
    ).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False)),
        alt.Y('slope:Q', scale=alt.Scale(zero=False)),
        text='Valley Name'
    )

    annotations2 = alt.Chart(src).mark_text(
        align='left',
        baseline='middle',
        fontSize = 14,
        dx = 7
    ).transform_filter(
        alt.FieldRangePredicate(field='drainage area', range=[0, 100000])
    ).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False)),
        alt.Y('slope:Q', scale=alt.Scale(zero=False)),
        text='Valley Name'
    )

    (chart + annotations1 + annotations2).configure_axis(grid=False)

    # %% [markdown]
    # ## Terrain characteristics of watersheds, hillslope + fluvial erosion area only

    # %% [markdown]
    # ### Calculate watershed statistics within erosion polygons only

    # %%
    limited_erosion_gdf = erosion_gdf.dissolve(by='name')

    limited_erosion_bytype_gdf = erosion_gdf.dissolve(by=['name', 'type'])

    # %%
    erosion_mean_results = limited_erosion_gdf.apply(
        (lambda row: dataset.rio.clip([row['geometry']]).to_dataframe().agg({
                'dtm': 'mean',
                'drainage area': 'max',
                'slope': 'mean',
                'aspect': 'mean',
                'curvature': 'mean',
                
                'ddem': 'sum',
                'ddem erosion': 'sum',
                'ddem hillslope': 'sum',
                'ddem fluvial': 'sum',
                'ddem negative': 'sum',
                'ddem erosion negative': 'sum',
                'ddem hillslope negative': 'sum',
                'ddem fluvial negative': 'sum',
                
            }
        )),
        axis='columns', result_type='expand'
    )
    limited_erosion_gdf[erosion_mean_results.columns] = erosion_mean_results

    # %%
    erosion_mean_results_bytype = limited_erosion_bytype_gdf.apply(
        (lambda row: dataset.rio.clip([row['geometry']]).to_dataframe().agg({
                'dtm': 'mean',
                'drainage area': 'max',
                'slope': 'mean',
                'aspect': 'mean',
                'curvature': 'mean',
                
                'ddem': 'sum',
                'ddem erosion': 'sum',
                'ddem hillslope': 'sum',
                'ddem fluvial': 'sum',
                'ddem negative': 'sum',
                'ddem erosion negative': 'sum',
                'ddem hillslope negative': 'sum',
                'ddem fluvial negative': 'sum',
                
            }
        )),
        axis='columns', result_type='expand'
    )
    limited_erosion_bytype_gdf[erosion_mean_results_bytype.columns] = erosion_mean_results_bytype

    # %%
    src = limited_erosion_gdf.drop(columns=['geometry']).reset_index()

    # src = src[src.name.isin(['Easton', 'Rainbow', 'Coleman', 'Deming', 'Mazama'])]

    chart = alt.Chart(src).mark_circle(size=100).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False), title='Drainage Area at Watershed Outlet'),
        alt.Y('slope:Q', scale=alt.Scale(zero=False), title='Mean Slope in Erosion Measurement Area'),
        # alt.Color('ddem erosion negative:Q', scale=alt.Scale(scheme='viridis'))
    )
    annotations1 = alt.Chart(src).mark_text(
        align='right',
        baseline='middle',
        fontSize = 10,
        dx = -7
    ).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False)),
        alt.Y('slope:Q', scale=alt.Scale(zero=False)),
        text='name'
    )

    (chart + annotations1).configure_axis(grid=False)

    # %%
    src = limited_erosion_bytype_gdf.drop(columns=['geometry']).reset_index()

    src = src.query("type == 'fluvial'")

    # src = src[src.name.isin(['Easton', 'Rainbow', 'Coleman', 'Deming', 'Mazama'])]

    chart = alt.Chart(src).mark_circle(size=100).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False), title='Drainage Area at Watershed Outlet'),
        alt.Y('slope:Q', scale=alt.Scale(zero=False), title='Mean Slope in Erosion Measurement Area'),
        # alt.Color('ddem erosion negative:Q', scale=alt.Scale(scheme='viridis'))
    )
    annotations1 = alt.Chart(src).mark_text(
        align='right',
        baseline='middle',
        fontSize = 10,
        dx = -7
    ).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False)),
        alt.Y('slope:Q', scale=alt.Scale(zero=False)),
        text='name'
    )

    (chart + annotations1).configure_axis(grid=False)

    # %%
    src = limited_erosion_bytype_gdf.drop(columns=['geometry']).reset_index()

    src = src.query("type == 'hillslope'")

    # src = src[src.name.isin(['Easton', 'Rainbow', 'Coleman', 'Deming', 'Mazama'])]

    chart = alt.Chart(src).mark_circle(size=100).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False), title='Drainage Area at Watershed Outlet'),
        alt.Y('slope:Q', scale=alt.Scale(zero=False), title='Mean Slope in Erosion Measurement Area'),
        # alt.Color('ddem erosion negative:Q', scale=alt.Scale(scheme='viridis'))
    )
    annotations1 = alt.Chart(src).mark_text(
        align='right',
        baseline='middle',
        fontSize = 10,
        dx = -7
    ).encode(
        alt.X('drainage area:Q', scale=alt.Scale(zero=False)),
        alt.Y('slope:Q', scale=alt.Scale(zero=False)),
        text='name'
    )

    (chart + annotations1).configure_axis(grid=False)

    # %%
    limited_erosion_gdf.reset_index().to_csv("outputs/terrain_attributes_erosionarea.csv", index=False)

    limited_erosion_bytype_gdf.reset_index().to_csv("outputs/terrain_attributes_erosionarea_bytype.csv", index=False)

    nhd_df.reset_index().to_csv("outputs/terrain_attributes_watershedarea.csv")


    process_gdf.reset_index(drop=True).drop(columns='id').to_csv("outputs/terrain_attributes_processpolygons.csv")

    # %%
    process_gdf

    # %%

    process_gdf.reset_index(drop=True).drop(columns='id').to_csv("outputs/terrain_attributes_processpolygons_maxdrainage.csv")