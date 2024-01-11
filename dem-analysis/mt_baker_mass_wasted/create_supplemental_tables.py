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
#     display_name: hsfm
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import os

# %%
if __name__ == "__main__":   
    modeling_powerlaw_data = pd.read_csv(
        'outputs/modeling_powerlaw_data.csv'
    )
    time_series_annualized_gross = pd.read_csv(
        'outputs/final_figures_data/time_series_annualized_gross.csv'
    )
    time_series_cumulative = pd.read_csv(
        'outputs/final_figures_data/time_series_cumulative.csv'
    )
    volumes_and_yields_per_valley = pd.read_csv(
        'outputs/final_figures_data/volumes_and_yields_per_valley.csv'
    )

    # %% [markdown]
    # # Gross Changes Table

    # %%
    time_series_annualized_gross['uncertainty'] = np.abs(time_series_annualized_gross['Annual Mass Wasted'] - time_series_annualized_gross['Lower CI'])
    time_series_annualized_gross = time_series_annualized_gross[['valley', 'process', 'type', 'start_time', 'end_time', 'Annual Mass Wasted', 'uncertainty']]
    time_series_annualized_gross = time_series_annualized_gross.sort_values(["valley", 'process', 'type', 'start_time', 'end_time'])
    time_series_annualized_gross['Annual Mass Wasted'] = np.round(time_series_annualized_gross['Annual Mass Wasted'], 2)
    time_series_annualized_gross['uncertainty'] = np.round(time_series_annualized_gross['uncertainty'], 2)
    time_series_annualized_gross

    # %% [markdown]
    # # Cumulative Changes Table

    # %%
    time_series_cumulative = time_series_cumulative[time_series_cumulative.type.isin(['fluvial', 'hillslope', 'all'])]
    time_series_cumulative['uncertainty'] = np.abs(time_series_cumulative['cumulative volume'] - time_series_cumulative['Lower CI'])
    time_series_cumulative = time_series_cumulative[['valley', 'type', 'end_time', 'cumulative volume', 'uncertainty']]
    time_series_cumulative['uncertainty'] = time_series_cumulative.apply(lambda row: row.uncertainty if row.end_time[:4] =='2015' else np.nan, axis=1)
    time_series_cumulative = time_series_cumulative.sort_values(['valley', 'type', 'end_time'])
    time_series_cumulative['cumulative volume'] = np.round(time_series_cumulative['cumulative volume'], 2)
    time_series_cumulative['uncertainty'] = np.round(time_series_cumulative['uncertainty'], 2)

    time_series_cumulative

    # %% [markdown]
    # # 10 Watershed Sediment Yields Table and Modeling Data

    # %%
    modeling_powerlaw_data['Uncertainty (ton/yr)'] = np.abs(modeling_powerlaw_data['Sediment Yield (ton/yr)'] - modeling_powerlaw_data['Lower CI sediment yield'])
    modeling_powerlaw_data['Uncertainty (ton/km²/yr)'] = np.abs(modeling_powerlaw_data['Sediment Yield (ton/km²/yr)'] - modeling_powerlaw_data['Lower CI sediment yield normalized'])

    # %%
    modeling_powerlaw_data = modeling_powerlaw_data[[
        'Valley Name',
        'Drainage area (square km)',
        'Channel slope',
        'Hillslope domain slope',
        'Glacial retreat area (km²)',
        'Nonigneous fraction',
        'Sediment Yield (ton/yr)',
        'Sediment Yield (ton/km²/yr)',
        'Uncertainty (ton/yr)',
        'Uncertainty (ton/km²/yr)'
    ]]

    modeling_powerlaw_data['Drainage area (square km)'] = np.round(modeling_powerlaw_data['Drainage area (square km)'], 2)
    modeling_powerlaw_data['Channel slope'] = np.round(modeling_powerlaw_data['Channel slope'], 2)
    modeling_powerlaw_data['Hillslope domain slope'] = np.round(modeling_powerlaw_data['Hillslope domain slope'], 2)
    modeling_powerlaw_data['Glacial retreat area (km²)'] = np.round(modeling_powerlaw_data['Glacial retreat area (km²)'], 2)
    modeling_powerlaw_data['Nonigneous fraction'] = np.round(modeling_powerlaw_data['Nonigneous fraction'], 2)
    modeling_powerlaw_data['Sediment Yield (ton/yr)'] = np.round(modeling_powerlaw_data['Sediment Yield (ton/yr)'], 2)
    modeling_powerlaw_data['Sediment Yield (ton/km²/yr)'] = np.round(modeling_powerlaw_data['Sediment Yield (ton/km²/yr)'], 2)
    modeling_powerlaw_data['Uncertainty (ton/yr)'] = np.round(modeling_powerlaw_data['Uncertainty (ton/yr)'], 2)
    modeling_powerlaw_data['Uncertainty (ton/km²/yr)'] = np.round(modeling_powerlaw_data['Uncertainty (ton/km²/yr)'], 2)

    # %%
    volumes_and_yields_per_valley['Uncertainty (1000 m³/yr)'] = np.abs(volumes_and_yields_per_valley['Annual Mass Wasted'] - volumes_and_yields_per_valley['Lower CI'])
    volumes_and_yields_per_valley['Sediment Yield (1000 m³/yr)'] = volumes_and_yields_per_valley['Annual Mass Wasted']
    volumes_and_yields_per_valley = volumes_and_yields_per_valley[volumes_and_yields_per_valley['named interval'] == 'bounding'][['Valley Name', 'Uncertainty (1000 m³/yr)', 'Sediment Yield (1000 m³/yr)']]
    volumes_and_yields_per_valley['Uncertainty (1000 m³/yr)'] = np.round(volumes_and_yields_per_valley['Uncertainty (1000 m³/yr)'], 2)
    volumes_and_yields_per_valley['Sediment Yield (1000 m³/yr)'] = np.round(volumes_and_yields_per_valley['Sediment Yield (1000 m³/yr)'], 2)

    volumes_and_yields_per_valley

    # %%
    modeling_powerlaw_data = modeling_powerlaw_data.merge(volumes_and_yields_per_valley, on='Valley Name')

    # %%
    modeling_powerlaw_data = modeling_powerlaw_data[[
        'Valley Name',
        'Drainage area (square km)',
        'Channel slope',
        'Hillslope domain slope',
        'Glacial retreat area (km²)',
        'Nonigneous fraction',
        'Sediment Yield (1000 m³/yr)',
        'Uncertainty (1000 m³/yr)',
        'Sediment Yield (ton/yr)',
        'Uncertainty (ton/yr)',
        'Sediment Yield (ton/km²/yr)',
        'Uncertainty (ton/km²/yr)',
    ]]


    modeling_powerlaw_data

    # %% [markdown]
    # # Save Completed Tables

    # %%
    if not os.path.exists("outputs/supplemental_tables"):
        os.mkdir("outputs/supplemental_tables")
    time_series_annualized_gross.to_csv("outputs/supplemental_tables/time_series_annualized_gross.csv")
    time_series_cumulative.to_csv("outputs/supplemental_tables/time_series_cumulative.csv")
    modeling_powerlaw_data.to_csv("outputs/supplemental_tables/modeling_powerlaw_data.csv")

    # %%
    src = time_series_annualized_gross.groupby(["valley", "type", "start_time", "end_time"]).sum()

    src = src.pivot_table(
        values=['Annual Mass Wasted',	'uncertainty'],
        index=['valley', 'start_time', 'end_time'],
        columns = ['type']
    )

    src.columns = src.columns.to_series().str.join('_')


    # %%
    src['sediment yield'] = -src['Annual Mass Wasted_negative'] - src['Annual Mass Wasted_positive']
    src['sediment delivery ratio'] = src['sediment yield']/-src['Annual Mass Wasted_negative']

    src = src.reset_index()

    # %%
    src['start_time'] = pd.to_datetime(src.reset_index().start_time)
    src['end_time'] = pd.to_datetime(src.reset_index().end_time)

    # %%
    src['sediment delivery ratio'] = np.round(src['sediment delivery ratio'],2)

    # %%
    src

    # %%
    import altair as alt

    # %%
    alt.Chart(src).mark_bar().encode(
        alt.X("start_time:T"),
        alt.X2("end_time:T"),
        alt.Y("sediment delivery ratio:Q")
    ).facet("valley:N")

    # %%
    src['net'] = -(src['Annual Mass Wasted_negative'] + src['Annual Mass Wasted_positive'])

    # %%
    src['Measurement period'] = ((src['start_time'] + (src['end_time'] - src['start_time']) / 2).dt.year).astype(str).apply(lambda x: x[:3] + "0's")

    src['Measurement period'] = src['Measurement period'].apply({
        "1950's": "1947-70",
        "1970's": "1970's",
        "1980's": "1980's",
        "2000's": "1990-2015"
    }.get)

    # %%
    src.sort_values('sediment delivery ratio')[['valley', 'start_time', 'end_time', 'sediment delivery ratio', 'net']]

    # %%
    net_erosion_data = src.sort_values('sediment delivery ratio')[['valley', 'start_time', 'end_time', 'sediment delivery ratio', 'net']].query("net > 0").reset_index(drop=True)['sediment delivery ratio']

    # %%
    net_erosion_data

    # %%
    net_erosion_data.mean(), net_erosion_data.median()

    # %%
    figure11 = alt.Chart(src).mark_point(size=50).encode(
        alt.X("net:Q", title ='Net Volumetric Change (10³ m³/yr)'),
        alt.Y("sediment delivery ratio:Q", title=['RSI', '(relative sediment balance index)']),
        alt.Shape("valley:N", title='Valley Name'),
        alt.Color("Measurement period:N", title=['Measurement', 'Period'])
    ).properties(width=200, height=200).configure_legend(
        titleFontSize=14, labelFontSize=14, orient='right'
    ).configure_axis(
        labelFontSize=14, titleFontSize=14
    )
    if not os.path.exists('outputs/final_figures'):
        os.mkdir('outputs/final_figures')
    figure11.save('outputs/final_figures/figure11.png')
    figure11
