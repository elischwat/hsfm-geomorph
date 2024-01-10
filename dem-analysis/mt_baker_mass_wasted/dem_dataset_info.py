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

dems_mosaic_path = os.path.join(BASE_PATH, "timesift/baker-ee-many/mixed_timesift/individual_clouds/final_products/dems_mosaiced/")

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
        glob.glob(os.path.join(params['inputs']['dems_path'], "*.tif"))
    ]
    valid_dem_dates = [date for date in all_dem_dates if date not in params['inputs']['TO_DROP']]
    valid_dem_dates_largearea = [date for date in all_dem_dates if date not in params['inputs']['TO_DROP_LARGER_AREA']]
    df = pd.DataFrame()
    df['Date'] = valid_dem_dates
    df['Larger Area'] = df['Date'].apply(lambda date: date in valid_dem_dates_largearea)
    df['Valley'] = valley
    dem_data_df = pd.concat([dem_data_df, df])

dem_data_df = pd.concat([dem_data_df, whole_mtn_data_manual])

dem_data_df['Date'] = dem_data_df['Date'].apply(lambda d: datetime.strptime(d, strip_time_format))   
dem_data_df['Description'] = dem_data_df['Date'].apply(lambda d: data_types.get(datetime.strftime(d, strip_time_format), default_data_type))
dem_data_df['Graph Start Date'] = dem_data_df['Date'].apply(lambda d: d - timedelta(days=230))
dem_data_df['Graph End Date'] = dem_data_df['Date'].apply(lambda d: d + timedelta(days=90))
# dem_data_df['Graph Start Date'] = dem_data_df['Date'].apply(lambda d: d - timedelta(days=30))
# dem_data_df['Graph End Date'] = dem_data_df['Date'].apply(lambda d: d + timedelta(days=30))
dem_data_df['Valley'] = dem_data_df['Valley'].apply(lambda x: x.title())
# -

dem_data_df

# +
alt.Chart(dem_data_df).mark_bar(
    opacity = 0.5
).encode(
    alt.X('Valley:N', axis=alt.Axis(labelAngle=0)),
    alt.Y('Graph Start Date:T'),
    alt.Y2('Graph End Date:T'),
    alt.Color('Description:N',
    legend=alt.Legend(
        orient='none',
        legendX=0, legendY=-40,
        direction='horizontal',
        titleAnchor='middle'
    )
    )
).properties(
    width=400,
    height=600

) + alt.Chart(dem_data_df).transform_filter(
    alt.FieldEqualPredicate(field='Larger Area', equal=True)
).mark_bar(
    color='black',
    filled=False
).encode(
    alt.X('Valley:N', axis=alt.Axis(labelAngle=0)),
    alt.Y('Graph Start Date:T'),
    alt.Y2('Graph End Date:T'),
    # alt.Color('Description:N')
).properties(
    width=400,
    height=600

)

# +
src = dem_data_df.query("Valley != 'Easton'")
src = src[
    ~((src.Valley == 'Whole Mountain') & (src['Graph Start Date'].dt.year == 1977))
]

src = src[
    ~((src.Valley == 'Whole Mountain') & (src['Graph Start Date'].dt.year == 1979))
]
(
    alt.Chart(src).mark_bar(
        opacity = 0.5
    ).encode(
        alt.X('Valley:N', axis=alt.Axis(labelAngle=-20)),
        alt.Y('Graph Start Date:T'),
        alt.Y2('Graph End Date:T'),
        alt.Color('Description:N',
        legend=alt.Legend(
            orient='top',
            # legendX=0, legendY=-40,
            # direction='horizontal',
            titleAnchor='middle',
            columns=1,
            symbolLimit=0,
            labelLimit=0
        )
        )
    ).properties(
        width=400,
        height=600

    ) + alt.Chart(src).transform_filter(
        alt.FieldEqualPredicate(field='Larger Area', equal=True)
    ).mark_bar(
        color='black',
        filled=False
    ).encode(
        alt.X('Valley:N', axis=alt.Axis(labelAngle=-20)),
        alt.Y('Graph Start Date:T'),
        alt.Y2('Graph End Date:T'),
        # alt.Color('Description:N')
    ).properties(
        width=400,
        height=600
    )
).configure_legend(labelFontSize=18, titleFontSize=18).configure_axis(labelFontSize=18, titleFontSize=18)
# -



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

alt.Chart(src).mark_bar(
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

# +
src = dem_data_df.query("Valley != 'Easton'")

src['Graph End Date'] = src['Graph End Date'].apply(lambda d: d + timedelta(days=90))


(alt.Chart(src).mark_bar(
    opacity = 0.5
).encode(
    alt.X('Valley:N', axis=alt.Axis(labelAngle=-20, tickCount=5)),
    alt.Y('Graph Start Date:T'),
    alt.Y2('Graph End Date:T'),
    alt.Color('Description:N',
    legend=alt.Legend(
        orient='top',
        # legendX=0, legendY=-40,
        # direction='horizontal',
        # titleAnchor='middle'
    )
    )
).properties(
    width=400,
    height=400

) + alt.Chart(src).transform_filter(
    alt.FieldEqualPredicate(field='Larger Area', equal=True)
).mark_bar(
    color='black',
    filled=False
).encode(
    alt.X('Valley:N', axis=alt.Axis(labelAngle=-20)),
    alt.Y('Graph Start Date:T', title='', axis=alt.Axis(tickCount=5)),
    alt.Y2('Graph End Date:T'),
    # alt.Color('Description:N')
).properties(
    width=400,
    height=400
)).configure_axis(labelFontSize=20, titleFontSize=20).configure_legend(labelFontSize=20, titleFontSize=20)
# -

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



dem_lowres_dataset = create_dataset(dems_mosaic_path)

# ## Plot all DEM mosaics

# +
bounds = dem_lowres_dataset['1977_09_27'].rio.bounds()

rows = 3
columns = 6

fig = plt.figure(figsize=(15,7))


items = list((dem_lowres_dataset.items()))
items.sort(key=lambda x: x[0])

for i, (key, raster) in enumerate(items):
    raster_clipped = raster.rio.clip_box(*bounds)
    raster_clipped = raster_clipped.rio.pad_box(*bounds)
    
    ax = plt.subplot(rows, columns, i + 1, aspect='auto')
    ax.imshow(raster_clipped.squeeze(), interpolation='none')
    ax.imshow(hillshade(raster_clipped.squeeze()), interpolation='none', cmap='gray_r', alpha=0.5)
    
    # ax.set_xlim(bounds[0],bounds[2])
    # ax.set_ylim(bounds[1],bounds[3])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(key.replace('.0',''),size=18)
    ax.set_facecolor('black')

fig.suptitle('Mount Baker DEM Gallery', fontsize=18)
fig.set_facecolor("w")
plt.tight_layout()
plt.savefig('dem_gallery_filtered_dems_by_date.png')
plt.show()

# -




# +
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
plt.savefig('dem_gallery_filtered_dems_by_date.png')
plt.show()

# -



axes

# +
vmin = np.nanmin(dem_lowres_dataset['2015_09_01'].values)
vmax = np.nanmax(dem_lowres_dataset['2015_09_01'].values)
print(vmin)
print(vmax)
bounds = dem_lowres_dataset['1977_09_27'].rio.bounds()

rows = 1
columns = 10



items = list((dem_lowres_dataset.items()))
items.sort(key=lambda x: x[0])


locations = list(range(0,10))



used_dates = dem_data_df['Date'].apply(lambda x: x.strftime(strip_time_format))
items = [i for i in items if i[0] in list(used_dates)]



###### NEW ###### 
fig, axes = plt.subplots(rows, columns, figsize=(44, 5))
###### NEW ###### 


for i, (key, raster) in enumerate(items):
    raster_clipped = raster.rio.clip_box(*bounds)
    raster_clipped = raster_clipped.rio.pad_box(*bounds)

    ###### NEW ###### 
    ax = axes[
        i
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
    ax.set_title(key.replace('.0','').replace('_', '/').split('/')[0],size=36)
    ax.set_facecolor('black')

# fig.suptitle('Mount Baker DEM Gallery', fontsize=36)
fig.set_facecolor("w")
# plt.tight_layout()
plt.savefig('dem_gallery_filtered_dems_by_date.png')
plt.show()


# +
vmin = np.nanmin(dem_lowres_dataset['2015_09_01'].values)
vmax = np.nanmax(dem_lowres_dataset['2015_09_01'].values)
print(vmin)
print(vmax)
bounds = dem_lowres_dataset['1977_09_27'].rio.bounds()

rows = 10
columns = 1



items = list((dem_lowres_dataset.items()))
items.sort(key=lambda x: x[0])


locations = list(range(0,10))



used_dates = dem_data_df['Date'].apply(lambda x: x.strftime(strip_time_format))
items = [i for i in items if i[0] in list(used_dates)]



###### NEW ###### 
fig, axes = plt.subplots(rows, columns, figsize=(5, 44))
###### NEW ###### 


for i, (key, raster) in enumerate(items):
    raster_clipped = raster.rio.clip_box(*bounds)
    raster_clipped = raster_clipped.rio.pad_box(*bounds)

    ###### NEW ###### 
    ax = axes[
        i
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
    ax.set_title(key.replace('.0','').replace('_', '/').split('/')[0],size=36)
    ax.set_facecolor('black')

# fig.suptitle('Mount Baker DEM Gallery', fontsize=36)
fig.set_facecolor("w")
# plt.tight_layout()
plt.savefig('dem_gallery_filtered_dems_by_date.png')
plt.show()

# -


