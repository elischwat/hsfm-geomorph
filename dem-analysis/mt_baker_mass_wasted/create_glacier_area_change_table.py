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

import pandas as pd
import geopandas as gpd
import glob
import os

BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
print(f"retrieved base path: {BASE_PATH}")

glacier_polys_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/**/glaciers.geojson"), recursive=True)

glaciers = pd.DataFrame()
for fn in glacier_polys_fns:
    glaciers = gpd.GeoDataFrame(pd.concat([glaciers, gpd.read_file(fn)]))
glaciers = glaciers.to_crs("EPSG:32610")

relavent_glaciers = [
    'Deming Glacier WA', 
    'Rainbow Glacier WA',
    'Park Glacier WA',
    'Thunder Glacier WA', 
    'Squak Glacier WA', 
    'Boulder Glacier WA',
    'Talum Glaciers WA', 
    'Easton Glacier WA', 
    'Coleman Glacier WA',
    'Roosevelt Glacier WA', 
    'Mazama Glacier WA'
]
relavent_dates = ['1947_09_14', '1977_09_27', '1979_10_06', '2015_09_01']

glaciers = glaciers[glaciers.Name.isin(relavent_glaciers)]
glaciers = glaciers[glaciers.year.isin(relavent_dates)]

glaciers = pd.concat([
    glaciers.query("year == '1979_10_06' and Name != 'Rainbow Glacier WA' and Name != 'Park Glacier WA'"),
    glaciers.query("year == '1977_09_27' and Name == 'Rainbow Glacier WA'"),
    glaciers.query("year == '1977_09_27' and Name == 'Park Glacier WA'"),
    glaciers.query("year == '1947_09_14'"),
    glaciers.query("year == '2015_09_01'"),
])

glaciers['area'] = glaciers.geometry.area

glaciers = glaciers.groupby(['Name', 'year']).sum(numeric_only=True).reset_index()[['Name', 'year', 'area']]

# +

glaciers['area difference'] = glaciers.groupby('Name')['area'].diff().reset_index(drop=True)
glaciers['area percent change'] = (glaciers['area difference'] / glaciers['area'].shift(1))*100

glaciers['area'] = glaciers['area']/1000000
glaciers['area difference'] = glaciers['area difference']/1000000

glaciers
# -

glaciers = glaciers.pivot(index='Name', columns='year', values=['area', 'area difference', 'area percent change']).round(decimals=3)

glaciers

glaciers.to_csv("outputs/final_figures_data/glacier_area.csv")
glaciers.to_pickle("outputs/glacier_area.pickle")


