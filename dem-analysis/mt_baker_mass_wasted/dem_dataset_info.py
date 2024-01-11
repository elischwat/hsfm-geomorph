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
import glob
import pandas as pd
import json 
import altair as alt
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import rioxarray as rix
from xrspatial import hillshade
import matplotlib.pyplot as plt

# %%
if __name__ == "__main__":   

    BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
    print(f"retrieved base path: {BASE_PATH}")

    # +
    inputs_dir = "inputs/"
    strip_time_format = "%Y_%m_%d"
    default_data_type = 'SfM, NAGAP aerial'
    data_types = {
        '1947_09_14': 'SfM, Earth Explorer aerial (low res)',
        '2013_09_13': 'SfM, EarthDEM Worldview satellite',
        '2015_09_01': 'Aerial LIDAR',
        '2019_10_11': 'SfM, EarthDEM Worldview satellite',
        
    }

    whole_mtn_data_manual = pd.DataFrame({
        'Date': ['1947_09_14', '1977_09_27', '1979_10_06', '2015_09_01'],
        'Larger Area': [True, True, True, True],
        'Valley': ['Whole Mountain', 'Whole Mountain', 'Whole Mountain', 'Whole Mountain']
    })

    # CHANGE THIS
    dems_mosaic_path = os.path.join(BASE_PATH, "hsfm-geomorph/data/dems_mosaiced")

    inputs_dict = {}
    for input_file in os.listdir(inputs_dir):
        with open(os.path.join(inputs_dir, input_file), 'r') as j:
            input_file
            params = json.loads(j.read())
            inputs_dict[input_file.replace("_inputs.json", "")] = params
    # -

    # # Plot DEM Timelines

    # +
    # list all DEMs in dems_path
    # remove the "TO_DROP" dates
    # assign type
    dem_data_df = pd.DataFrame()
    for valley, params in inputs_dict.items():
        all_dem_dates = [Path(f).stem for f in 
            glob.glob(os.path.join(BASE_PATH, params['inputs']['dems_path'], "*.tif"))
        ]
        valid_dem_dates = [date for date in all_dem_dates if date not in params['inputs']['TO_DROP']]
        valid_dem_dates_largearea = [date for date in all_dem_dates if date not in params['inputs']['TO_DROP_LARGER_AREA']]
        df = pd.DataFrame()
        df['Date'] = valid_dem_dates
        df['Larger Area'] = df['Date'].apply(lambda date: date in valid_dem_dates_largearea)
        df['Valley'] = valley
        dem_data_df = pd.concat([
            dem_data_df, df])


    # %%
    dem_data_df = pd.concat([dem_data_df, whole_mtn_data_manual])

    dem_data_df['Date'] = dem_data_df['Date'].apply(lambda d: datetime.strptime(d, strip_time_format))   
    dem_data_df['Description'] = dem_data_df['Date'].apply(lambda d: data_types.get(datetime.strftime(d, strip_time_format), default_data_type))
    dem_data_df['Graph Start Date'] = dem_data_df['Date'].apply(lambda d: d - timedelta(days=230))
    dem_data_df['Graph End Date'] = dem_data_df['Date'].apply(lambda d: d + timedelta(days=90))
    # dem_data_df['Graph Start Date'] = dem_data_df['Date'].apply(lambda d: d - timedelta(days=30))
    # dem_data_df['Graph End Date'] = dem_data_df['Date'].apply(lambda d: d + timedelta(days=30))
    dem_data_df['Valley'] = dem_data_df['Valley'].apply(lambda x: x.title())
    # -



    # %%

    # +
    src = dem_data_df.query("Valley != 'Easton'")
    src = src[
        ~((src.Valley == 'Whole Mountain') & (src['Graph Start Date'].dt.year == 1977))
    ]

    src = src[
        ~((src.Valley == 'Whole Mountain') & (src['Graph Start Date'].dt.year == 1979))
    ]

    src['Graph End Date'] = src['Date'].apply(lambda d: d + timedelta(days=270))

    # -

    src = src[['Valley',
    'Graph Start Date',
    'Graph End Date',
    'Description']].rename(columns={'Valley':'Watershed'})
    src['Watershed'] = src['Watershed'].apply(lambda x: '10 Watersheds' if x == 'Whole Mountain' else x)

    figure2b = alt.Chart(src).mark_bar(
        opacity = 0.5
    ).encode(
        alt.Y('Watershed:N', axis=alt.Axis(labelAngle=0), sort=['Whole Mountain']),
        alt.X('Graph Start Date:T', title='Year',),
        alt.X2('Graph End Date:T'),
        alt.Color('Description:N',
            legend=alt.Legend(
                title=None,
                orient='top',
                titleAnchor='middle',
                symbolLimit=0,
                labelLimit=0
            )
        )
    ).configure_legend(
        labelFontSize=18, 
        titleFontSize=18
    ).configure_axis(
        labelFontSize=18, 
        titleFontSize=18
    ).properties(
        width=800, 
        height=200
    )

    if not os.path.exists("outputs/final_figures/"):
        os.makedirs("outputs/final_figures/")
    figure2b.save("outputs/final_figures/figure2b.png")

    # %%
    figure2b


    # %%
    # # Plot DEM Gallery
    def create_dataset(individual_clouds_path):
        """
        Depends on files being organized as is the output of the TimesiftPipeline classs.
        individual_clouds_path (str): Path ending with "individual_clouds"
        tif_name (str): type of file to use. options include 'dem', 'dod', and 'orthomosaic'
        """
        all_files = glob.glob(os.path.join(individual_clouds_path, "*.tif"), recursive=True)
        d = {}
        for f in all_files:
            tif_name = os.path.split(f)[-1].replace(".tif", "")
            print(f'Reading {f}')
            d[tif_name] = rix.open_rasterio(
                f, 
                masked=True
            )
            d[tif_name] = d[tif_name].rio.reproject(resolution=(30,30), dst_crs = d[tif_name].rio.crs)
        return d




    # %%

    dem_lowres_dataset = create_dataset(dems_mosaic_path)
    vmin = np.nanmin(dem_lowres_dataset['2015_09_01'].values)
    vmax = np.nanmax(dem_lowres_dataset['2015_09_01'].values)
    print(vmin)
    print(vmax)
    bounds = dem_lowres_dataset['1977_09_27'].rio.bounds()

    rows = 2
    columns = 5

    fig = plt.figure(figsize=(10, 24))


    items = list((dem_lowres_dataset.items()))
    items.sort(key=lambda x: x[0])

    #row col
    locations = [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 2],
        [1, 2],
        [0, 3],
        [1, 3],
        [0, 4],
        [1, 4],
    ]

    locations = [
        [0, 0],
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [1, 0],
        [1, 1],
        [1, 2],
        [1, 3],
        [1, 4],

    ]



    used_dates = dem_data_df['Date'].apply(lambda x: x.strftime(strip_time_format))
    items = [i for i in items if i[0] in list(used_dates)]



    ###### NEW ###### 
    fig, axes = plt.subplots(rows, columns, figsize=(22, 10))
    ###### NEW ###### 


    for i, (key, raster) in enumerate(items):
        raster_clipped = raster.rio.clip_box(*bounds)
        raster_clipped = raster_clipped.rio.pad_box(*bounds)

        ###### NEW ###### 
        ax = axes[
            locations[i][0], locations[i][1]
        ]
        ###### NEW ###### 
        
        ###### OLD ###### 
        # ax = plt.subplot(rows, columns, i + 1, aspect='auto')

        
        ax.imshow(raster_clipped.squeeze(), interpolation='none', vmin=vmin, vmax=vmax)
        ax.imshow(hillshade(raster_clipped.squeeze()), interpolation='none', cmap='gray_r', alpha=0.5)
        
        # ax.set_xlim(bounds[0],bounds[2])
        # ax.set_ylim(bounds[1],bounds[3])
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(key.replace('.0','').replace('_', '/'),size=36)
        ax.set_facecolor('black')

    fig.suptitle('Mount Baker DEM Gallery', fontsize=36)
    fig.set_facecolor("w")
    # plt.tight_layout()

    if not os.path.exists("outputs/final_figures/"):
        os.makedirs("outputs/final_figures/")
    plt.savefig("outputs/final_figures/figure2a.png")

    plt.show(block=False)