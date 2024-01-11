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
#     display_name: Python 3.8.6 ('hsfm-test')
#     language: python
#     name: python3
# ---

# %%
import geopandas as gpd
import pandas as pd
import contextily as ctx
import fiona
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":   
    # %%
    BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
    print(f"retrieved base path: {BASE_PATH}")

    # %% [markdown]
    # # Open Terrace Erosion Data

    # %%
    terrace_df = pd.read_excel(
        os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/terrace_erosion_reanalysis/2020wr028389-sup-0002-data set si-s01.xlsx")
    )

    # %%
    terrace_df = terrace_df.query("basin == 'Nooksack'")
    terrace_gdf = gpd.GeoDataFrame(
        terrace_df,
        geometry=gpd.points_from_xy(terrace_df['lon'], terrace_df['lat']),
        crs='EPSG:4326'
    )

    # %%
    terrace_gdf.to_file(
        os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/terrace_erosion_reanalysis/nooksack_terrace_data.geojson"), 
        driver='GeoJSON'
    )

    # %% [markdown]
    # # Open Watershed Data

    # %%
    nooksack_forks = [
        'Lower North Fork Nooksack River',
        'Upper North Fork Nooksack River',
        'South Fork Nooksack River',
        'Middle Fork Nooksack River'
    ]

    # %%
    fiona.listlayers(
        os.path.join(BASE_PATH, "hsfm-geomorph/data/NHDPLUS_H_1711_HU4_GDB/NHDPLUS_H_1711_HU4_GDB.gdb")
    )[65]

    # %%
    wsheds_gdf = gpd.read_file(
        os.path.join(BASE_PATH, "hsfm-geomorph/data/NHDPLUS_H_1711_HU4_GDB/NHDPLUS_H_1711_HU4_GDB.gdb"),
        layer=65
    ).to_crs('EPSG:32610')
    wsheds_gdf = wsheds_gdf[wsheds_gdf['Name'].isin(nooksack_forks)]

    # %% [markdown]
    # # Retrieve USGS streamgage locations

    # %%
    station_df = pd.DataFrame({
        'site name': [
            "NF NOOKSACK RIVER BL CASCADE CREEK NR GLACIER, WA",
            "MF NOOKSACK RIVER NEAR DEMING, WA",
            "NOOKSACK RIVER AT NORTH CEDARVILLE, WA",
            "SF NOOKSACK RIVER AT SAXON BRIDGE, WA",
            "NF NOOKSACK RIVER NEAR DEMING, WA"
        ],
        'lat': [
            48.90595739,
            48.7792828,
            48.84178209,
            48.67761256,
            48.8731746
        ],
        'lon': [
            -121.8443104,
            -122.1065434,
            -122.2943258,
            -122.1665454,
            -122.1501529 
        ]
    })

    # %%

    station_gdf = gpd.GeoDataFrame(
        station_df,
        geometry=gpd.points_from_xy(station_df['lon'], station_df['lat']),
        crs='EPSG:4326'
    )

    # %% [markdown]
    # # Map stuff

    # %%
    terrace_gdf = terrace_gdf.to_crs(wsheds_gdf.crs)
    station_gdf = station_gdf.to_crs(wsheds_gdf.crs)

    # %%
    # %matplotlib inline

    # %%
    import glob
    streamstats_watersheds_fns = glob.glob(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/streamstats_watersheds/*.geojson"))

    gdf = gpd.GeoDataFrame()
    for f in streamstats_watersheds_fns:
        new_data = gpd.read_file(f)
        new_data['Valley Name'] = f.split("/")[-1].split(".geojson")[0]
        gdf = gpd.GeoDataFrame(pd.concat([gdf, new_data]))

    nhd_df = gdf[gdf.geometry.type != "Point"]
    nhd_df = nhd_df.to_crs(wsheds_gdf.crs)

    # %%
    nhd_df['Valley Name'] = nhd_df['Valley Name'].str.capitalize()

    # %%
    wsheds_gdf.Name.unique()

    # %%
    wsheds_gdf['Short Name'] = wsheds_gdf['Name'].apply({
        'Upper North Fork Nooksack River': 'North Fork',
        'South Fork Nooksack River': 'South Fork',
        'Middle Fork Nooksack River': 'Middle Fork',
        'Lower North Fork Nooksack River': 'North Fork'
    }.get)

    # %% [markdown]
    # To Do: 
    #
    # * Combine the two north fork polygons
    # * Remove gages that we aren't using (the one in Lower North Fork watershed )
    # * Give short names to the watersheds
    # * Add lines for the Nooksack river stream location (Where do i get this?)

    # %%
    import matplotlib.patheffects as pe
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    f, ax = plt.subplots(1, figsize=(9, 9))
    nhd_src = nhd_df[nhd_df['Valley Name'].isin(['Coleman', 'Mazama', 'Thunder', 'Deming'])]

    nhd_src.apply(lambda x: ax.annotate(text=x['Valley Name'], xy=(x.geometry.centroid.coords[0][0]+5000, x.geometry.centroid.coords[0][1]), ha='left', fontsize=10, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]), axis=1)
    nhd_src.plot(ax=ax, color='none', edgecolor='blue', linewidth=2)

    wsheds_gdf.plot(ax=ax, color='none', edgecolor='black', linewidth=2)
    wsheds_gdf.apply(lambda x: ax.annotate(text=x['Short Name'], xy=x.geometry.centroid.coords[0], ha='center', fontsize=12, path_effects=[pe.withStroke(linewidth=4, foreground="white")]), axis=1)

    terrace_gdf.plot(ax=ax)
    station_gdf.plot(ax=ax, color='red')

    handles = [
        mpatches.Patch(facecolor = 'none', edgecolor = 'blue', linewidth=2, label='Watersheds (this study)'),
        mpatches.Patch(facecolor = 'none', edgecolor = 'black', linewidth=2, label='Watersheds (Nooksack River)'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='none', markerfacecolor='blue', label='Terrace Erosion Site', markersize=8),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='none', markerfacecolor='red', label='Stream Gage', markersize=8)
    ]

    plt.legend(handles=handles)

    ctx.add_basemap(ax, crs = wsheds_gdf.crs)
    plt.ticklabel_format(style='plain')
    plt.show(block=False)

    # %%
    wsheds_gdf_just3 = pd.concat([
        wsheds_gdf[wsheds_gdf['Short Name'] != 'North Fork'][['Name', 'Short Name', 'geometry']],
        gpd.GeoDataFrame({
        'Name' : ['North Fork Nooksack River'],
        'Short Name': ['North Fork'],
        'geometry': [wsheds_gdf.loc[wsheds_gdf['Name'] == 'Upper North Fork Nooksack River'].geometry.iloc[0].union(
            wsheds_gdf.loc[wsheds_gdf['Name'] == 'Lower North Fork Nooksack River'].geometry.iloc[0]
        )]
    })
    ])

    wsheds_gdf_just3 = wsheds_gdf_just3.set_crs(wsheds_gdf.crs) 

    stations_gdf_just4 = station_gdf.iloc[:-1]

    # %%
    import cartopy

    # %%
    cartopy.feature.RIVERS

    # %%
    streamlines_gdf = gpd.read_file(os.path.join(BASE_PATH, "hsfm-geomorph/data/mt_baker_mass_wasted/ECY_WAT_NHDWAMajor.gdb"), layer=2).to_crs(wsheds_gdf_just3.crs)

    # %%
    streamlines_gdf = streamlines_gdf.cx[548965.5616015878: 610837.19530237, 5377715.576250239:5434687.776431516]
    streamlines_gdf = streamlines_gdf[streamlines_gdf.GNIS_Name.isin([
        'North Fork Nooksack River', 'Middle Fork Nooksack River',
        'Nooksack River', 'South Fork Nooksack River',
    ])]

    # %%
    import matplotlib.patheffects as pe
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    f, ax = plt.subplots(1, figsize=(9, 9), dpi=300)

    streamlines_gdf.plot(ax=ax, color='blue', linewidth=1, zorder=1, alpha=0.6)

    nhd_src = nhd_df[nhd_df['Valley Name'].isin(['Coleman', 'Mazama', 'Thunder', 'Deming'])]

    nhd_src.apply(lambda x: ax.annotate(text=x['Valley Name'], xy=(x.geometry.centroid.coords[0][0]+3500, x.geometry.centroid.coords[0][1]), ha='left', fontsize=10, path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]), axis=1)
    nhd_src.plot(ax=ax, color='none', edgecolor='purple', linewidth=2)

    wsheds_gdf_just3.plot(ax=ax, color='none', edgecolor='black', linewidth=2)
    wsheds_gdf_just3.apply(lambda x: ax.annotate(text=x['Short Name'], xy=(x.geometry.centroid.coords[0][0], x.geometry.centroid.coords[0][1]-1000), ha='center', fontsize=11, path_effects=[pe.withStroke(linewidth=4, foreground="white")]), axis=1)

    terrace_gdf.plot(ax=ax, color='red')
    stations_gdf_just4.plot(ax=ax, color='green', markersize=100)

    handles = [
        mpatches.Patch(facecolor = 'none', edgecolor = 'purple', linewidth=2, label='Watershed boundaries\n(this study)'),
        mpatches.Patch(facecolor = 'none', edgecolor = 'black', linewidth=2, label='Watershed boundaries\n(Nooksack River Forks, HUC10)'),
        Line2D([4], [0], color='blue', label='Nooksack River'),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='none', markerfacecolor='red', label='Terrace erosion site', markersize=8),
        Line2D([0], [0], marker='o', color='none', markeredgecolor='none', markerfacecolor='green', label='Stream gage', markersize=8),
        
    ]

    ax.annotate(
        text='Main Stem',
        xy=(552750, 5415000),
        ha='center',
        fontsize=11,
        path_effects=[pe.withStroke(linewidth=4,foreground="white")]
    )

    plt.legend(handles=handles, ncols=2)

    ctx.add_basemap(ax, crs = wsheds_gdf_just3.crs)
    plt.ticklabel_format(style='plain')
    plt.xticks(
        [560000, 580000, 600000],
        ['560000E', '580000E', '600000E']
    )
    plt.yticks(
        [5380000, 5400000, 5420000],
        ['5380000N', '5400000N', '5420000N']
    )
    plt.xlim(548965.5616015878, 610837.19530237)
    plt.ylim(5377715.576250239, 5434687.776431516)
    plt.show(block=False)

    # %% [markdown]
    # # Calculate total terrace erosion in Nooksack River

    # %%
    print("Nooksack River (whole)")
    print(round(terrace_gdf['bluff_erosion_vol_m3_per_yr'].sum()))
    print(round(terrace_gdf['bluff_erosion_vol_m3_per_yr_lo'].sum()))
    print(round(terrace_gdf['bluff_erosion_vol_m3_per_yr_hi'].sum()))
    for idx, row in wsheds_gdf.iterrows():
        print(row['Name'])
        local_terrace_erosion = terrace_gdf[terrace_gdf.within(
                row.geometry
            )]
        print(round(local_terrace_erosion['bluff_erosion_vol_m3_per_yr'].sum()))
        print(round(local_terrace_erosion['bluff_erosion_vol_m3_per_yr_lo'].sum()))
        print(round(local_terrace_erosion['bluff_erosion_vol_m3_per_yr_hi'].sum()))
