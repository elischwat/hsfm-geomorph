# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)

rgi_file = '/Users/elischwat/Development/dem_differencing/data/raw/glacier_polygons/00_rgi60/02_rgi60_WesternCanadaUS/02_rgi60_WesternCanadaUS.shp'
aois_file = 'aois.geojson'
imagery_file = '~/Downloads/glacier_names_pids.csv'

imagery_df = pd.read_csv(imagery_file)
rgi_gdf = gpd.read_file(rgi_file)
aois_gdf = gpd.read_file(aois_file)

# Don't think I need these

# +
# mt_rainier_glaciers = ['Russel','North Mowich','Fleet','Edmunds','South Mowich','Puyallup','Tahoma','South Tahoma','Van Trump','Success','Kautz','Pyramid','Wilson','Carbon','Winthrop','InterGlacier','Emmons','Summit','Sarvent','Fryingpan','Ohanapecosh','Whitman','Ingraham','Cowlitz','Paradise-Stevens','Nisqually']
# mt_baker_glaciers = ['Mazama','Hadley','Roosevelt','Coleman','Thunder','Deming','Easton','Squak','Tatum','Boulder','Park','Rainbow','Sholes']
# glacier_peak_glaciers = ['Chocolate','Cool','Dusty','Ermine','Honeycomb','Kennedy','Milk Lake','North Guardian','Ptarmigan','Scimiatr','Sitkum','Suiattle','Vista','White Chuck','White River']
# -

# ### Join the RGI polygons and AOIs polygons to filter out glacier polygons outside of our AOIs

joined = gpd.sjoin(rgi_gdf, aois_gdf)

# Number of RGI polygons that intersect our AOIs

len(joined)

# Number of RGI polyons that intersect particular AOIs

len(joined[joined.name == "Mt. Rainier"]), len(joined[joined.name == "Mt. Baker"]), len(joined[joined.name == "Glacier Peak"])

joined.head()

# ### Plot (Messy) AOI polygons

fig, axes = plt.subplots(1,3, figsize=(12,8))
aois_gdf[aois_gdf.name == "Mt. Rainier"].plot(ax = axes[0])
aois_gdf[aois_gdf.name == "Mt. Baker"].plot(ax = axes[1])
aois_gdf[aois_gdf.name == "Glacier Peak"].plot(ax = axes[2])
plt.tight_layout()
plt.show()

# ### Plot RGI Polygons that Intersect the AOIs

fig, axes = plt.subplots(1,3, figsize=(12,8))
joined[joined.name == "Mt. Rainier"].plot(ax = axes[0])
joined[joined.name == "Mt. Baker"].plot(ax = axes[1])
joined[joined.name == "Glacier Peak"].plot(ax = axes[2])
plt.tight_layout()
plt.show()

# ### Create geodataframe from the bounding boxes in imagery_df

from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(imagery_df.Longitude, imagery_df.Latitude)]
imagery_gdf = gpd.GeoDataFrame(imagery_df, crs="EPSG:4326", geometry=geometry)

imagery_in_aois = gpd.sjoin(aois_gdf, imagery_gdf, op='intersects', how='inner')

rainier_images = imagery_in_aois[imagery_in_aois.name == "Mt. Rainier"]
baker_images = imagery_in_aois[imagery_in_aois.name == "Mt. Baker"]
glacierpeak_images = imagery_in_aois[imagery_in_aois.name == "Glacier Peak"]

len(rainier_images), len(baker_images), len(glacierpeak_images)

rainier_images.Date.value_counts().sort_values()

baker_images.Date.value_counts().sort_values()

glacierpeak_images.Date.value_counts().sort_values()


