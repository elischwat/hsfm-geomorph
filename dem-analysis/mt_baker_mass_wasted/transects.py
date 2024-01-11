# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3.8.6 ('hsfm-test-backup')
#     language: python
#     name: python3
# ---

import geopandas as gpd
import shapely
import shapely
import rasterio
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append("..")
import os
import datetime
import profiling_tools
import altair as alt
from altair import datum
import json
alt.data_transformers.disable_max_rows()

if __name__ == "__main__":   

    # # Inputs
    # Provide:
    # - Input file path to file with cross section lines/polygons to extract low points/stream profile from
    # - Output file path where low points will be saved
    # - Input directory path to location of DEMs
    # - Parameter `LINE_COMPLEXITY` which is the number of points that each cross-section line is split into. `LINE_COMPLEXITY` elevation points will be extracted from the DEM for each cross section line

    # If you use the arg, you must run from CLI like this
    #
    # ```
    # HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_mazama.html
    # ```

    BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
    print(f"retrieved base path: {BASE_PATH}")

    # +

    # Or set an env arg:
    if os.environ.get('HSFM_GEOMORPH_INPUT_FILE'):
        json_file_path = os.environ['HSFM_GEOMORPH_INPUT_FILE']
    else:
        json_file_path = 'inputs/rainbow_inputs.json'
    # -

    with open(json_file_path, 'r') as j:
        params = json.loads(j.read())

    params

    # +
    TO_DROP = params['inputs']['TO_DROP']
    input_transects_file = os.path.join(BASE_PATH, params['transects']['input_transects_file'])
    input_dems_path = os.path.join(BASE_PATH, params['inputs']['dems_path'])
    glacier_polygons_file = os.path.join(BASE_PATH, params['inputs']['glacier_polygons_file'])
    LINE_COMPLEXITY = params['transects']['line_complexity']
    raster_fns = glob.glob(os.path.join(input_dems_path, "*.tif"))

    strip_time_format = params['inputs']['strip_time_format']

    reference_dem_date = datetime.datetime.strptime(
        params['inputs']['reference_dem_date'], 
        strip_time_format
    )

    # +

    raster_fns = [fn for fn in raster_fns if Path(fn).stem not in TO_DROP]
    raster_fns
    # -

    # # Extract profiles from DEMs 
    #
    # Along each cross-section, extract point with lowest elevation and calculate "path distance", the distance from the furthest downstream cross section line.

    # +
    # read cross sections file into GeoDataframe
    gdf = gpd.read_file(input_transects_file)
    # Increase the number of points in each line
    gdf.geometry = gdf.geometry.apply(lambda g: profiling_tools.increase_line_complexity(g, LINE_COMPLEXITY))
    # Get all points from the cross section lines and create a row for each point. 
    gdf['coords'] = gdf.geometry.apply(lambda x: list(x.coords))
    crs = gdf.crs
    gdf = gpd.GeoDataFrame(pd.DataFrame(gdf).explode('coords', ignore_index=True))
    # Make the coords column a shapely.geometry.Point type and drop the cross section geometries which we no longer need.
    gdf['coords'] = gdf['coords'].apply(shapely.geometry.Point)
    gdf.drop(columns=["geometry"])

    combined_gdf = gpd.GeoDataFrame()

    for raster in raster_fns:
        print(raster)
        # Extract an elevation value for each point
        with rasterio.open(raster) as src:
            new_gdf = gdf.copy()
            new_gdf['elevation'] = pd.Series([sample[0] for sample in src.sample(new_gdf["coords"].apply(lambda x: (x.xy[0][0], x.xy[1][0])))])
            new_gdf['elevation'] = new_gdf['elevation'].apply(lambda x: np.nan if x == src.nodata else x)

        # Convert file name to datetime as per the provided format
        date = datetime.datetime.strptime(Path(raster).stem, strip_time_format)
        new_gdf['time'] = date

        # Set the geometry to the coords to calculate "path distance"    
        combined_gdf = pd.concat([combined_gdf, new_gdf])
    combined_gdf.crs = crs
    combined_gdf

    # +

    for key, group in combined_gdf.groupby(["id", "time"]):
        group.geometry = group['coords']
        group['path_distance'] = pd.Series(group.distance(group.shift(1)).fillna(0)).cumsum()
        new_gdf = pd.concat([new_gdf, group])
    new_gdf.crs = crs
    # -

    # # Mark points as (non)glacial

    glaciers_gdf = gpd.read_file(glacier_polygons_file)
    glaciers_gdf = glaciers_gdf.to_crs(new_gdf.crs)
    glaciers_gdf['time'] = glaciers_gdf['year'].apply(lambda d: datetime.datetime.strptime(d, strip_time_format))

    new_gdf['glacial'] = new_gdf.apply(
        lambda row: any(glaciers_gdf.loc[glaciers_gdf['time'] == row["time"], 'geometry'].apply(lambda g: g.contains(row['coords']))),
        axis=1
    )

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).mark_line().encode(
        alt.X("path_distance:Q", scale=alt.Scale(zero=False)),
        alt.Y("elevation:Q", scale=alt.Scale(zero=False)),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 200,
        height = 200
    ).facet(
        row="id:O"
    ).resolve_scale(
        x="independent",
        y="independent"
    ).configure_legend(
        titleColor='black', 
        titleFontSize=12, 
        labelFontSize=16, 
        symbolStrokeWidth=4
    )

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).transform_window(
        rolling_mean='mean(elevation)',
        frame=[-8, 8],
        groupby=["id:O"]
    ).mark_line().encode(
        alt.X("path_distance:Q", scale=alt.Scale(zero=False)),
        alt.Y("rolling_mean:Q", scale=alt.Scale(zero=False)),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 200,
        height = 200
    ).facet(
        row="id:O"
    ).resolve_scale(
        x="independent",
        y="independent"
    ).configure_legend(
        titleColor='black', 
        titleFontSize=12, 
        labelFontSize=16, 
        symbolStrokeWidth=4
    )

    # ## Deming transect 1

    # +

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    src = src.query("id == 1")
    src = src.query("path_distance < 80")
    alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).transform_window(
        rolling_mean='mean(elevation)',
        frame=[-3, 3],
        groupby=["id:O"]
    ).mark_line().encode(
        alt.X("path_distance:Q", title="Distance (m)", scale=alt.Scale(zero=False, domain=[0,80])),
        alt.Y("rolling_mean:Q", title="Elevation (m)", scale=alt.Scale(zero=False, domain=[1140, 1160])),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 200,
        height = 100
    ).configure_legend(
        titleColor='black', 
        titleFontSize=12, 
        labelFontSize=16, 
        symbolStrokeWidth=4
    )
    # -

    # ## Deming transect 2

    # +

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    src = src.query("id == 2")
    alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).transform_window(
        rolling_mean='mean(elevation)',
        frame=[-3, 3],
        groupby=["id:O"]
    ).transform_filter(
        alt.datum.path_distance > 2
    ).transform_filter(
        alt.datum.path_distance < 117
    ).mark_line().encode(
        alt.X("path_distance:Q", title="Distance (m)", scale=alt.Scale(zero=False, domain=[0,120])),
        alt.Y("rolling_mean:Q", title="Elevation (m)", scale=alt.Scale(zero=False, domain=[1060, 1200])),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 171.5,
        height = 200
    ).configure_legend(
        titleColor='black', 
        titleFontSize=12, 
        labelFontSize=16, 
        symbolStrokeWidth=4
    )
    # -

    # ## Rainbow transect 1
    #

    # +

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    src = src.query("id == 1")
    tran1 = alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).transform_window(
        rolling_mean='mean(elevation)',
        frame=[-3, 3],
        groupby=["id:O"]
    ).transform_filter(
        alt.datum.path_distance > 2
    ).transform_filter(
        alt.datum.path_distance < 400
    ).mark_line().encode(
        alt.X("path_distance:Q", title="Distance (m)", scale=alt.Scale(zero=False, domain=[0,400], nice=False)),
        alt.Y("rolling_mean:Q", title="Elevation (m)", scale=alt.Scale(zero=False, domain=[1100, 1350], nice=False)),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 320,
        height = 200
    )
    # -

    # ## Rainbow transect 2

    # +

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    src = src.query("id == 2")
    tran2 = alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).transform_window(
        rolling_mean='mean(elevation)',
        frame=[-5, 5],
        groupby=["id:O"]
    ).transform_filter(
        alt.datum.path_distance > 2
    ).transform_filter(
        alt.datum.path_distance < 110
    ).mark_line().encode(
        alt.X("path_distance:Q", title="Distance (m)", scale=alt.Scale(zero=False, domain=[0,110], nice=False)),
        alt.Y("rolling_mean:Q", title="Elevation (m)", scale=alt.Scale(zero=False, domain=[1290, 1330], nice=False)),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 325,
        height = 200
    )
    # -

    (tran1 | tran2).configure_legend(
        titleColor='black', 
        titleFontSize=12, 
        labelFontSize=16, 
        symbolStrokeWidth=4
    ).configure_axis(
        titleColor='black', 
        titleFontSize=14, 
        labelFontSize=16
    )

    # +
    width = 320, 
    height = 200

    width_distance = 65
    height_distance = 40
    # -

    320 * 40/65

    # ## Rainbow transect 3

    # +

    src = new_gdf[[ "time", "path_distance", "elevation", "id", "glacial"]].reset_index()
    src['time'] = src['time'].apply(lambda x: x.strftime("%Y-%m-%d"))
    src = src.query("id == 3")
    alt.Chart(
        src
    ).transform_filter(
        datum.glacial==False
    ).transform_window(
        rolling_mean='mean(elevation)',
        frame=[-5, 5],
        groupby=["id:O"]
    ).transform_filter(
        alt.datum.path_distance > 3
    ).transform_filter(
        alt.datum.path_distance < 315
    ).mark_line().encode(
        alt.X("path_distance:Q", title="Distance (m)", scale=alt.Scale(zero=False, 
        domain=[0,325], 
        nice=False)),
        alt.Y("rolling_mean:Q", title="Elevation (m)", scale=alt.Scale(zero=False, 
        domain=[1120, 1280], 
        nice=False)),
        alt.Color("time:O", scale=alt.Scale(scheme='turbo')),
    ).properties(
        width = 325,
        height = 200
    ).configure_legend(
        titleColor='black', 
        titleFontSize=12, 
        labelFontSize=16, 
        symbolStrokeWidth=4
    ).configure_axis(
        titleColor='black', 
        titleFontSize=14, 
        labelFontSize=16
    )
    # -


