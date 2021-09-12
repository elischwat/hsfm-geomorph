# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import altair as alt
import glob
import json
import pandas as pd


# +
def get_stats(fn):
    with open(fn) as src:
        y = json.load(src)
        return y['after_filt']['nmad']
#     (
#             y['after_filt']['nmad']
#             y['before']['nmad'], 
#             y['before_filt']['nmad'],
#             y['after']['nmad'],
            
#         )


# +
df = pd.DataFrame({
    'filename': pd.Series(glob.glob("/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/**/**/1/pc_align/*/*.json"))
})
df_first_iteration = pd.DataFrame({
    'filename': pd.Series(glob.glob("/data2/elilouis/mt_baker_timesift_cam_calib_highres/individual_clouds/**/**/0/pc_align/*/*.json"))
})
df['run'] = 'nagap images'

df['date'] = df['filename'].apply(lambda fn: fn.split('/')[5])
df['cluster'] = df['filename'].apply(lambda fn: fn.split('/')[6])
df['iteration'] = df['filename'].apply(lambda fn: fn.split('/')[7])
df['stats'] = df['filename'].apply(lambda fn: get_stats(fn))
df['stats_first_iteration'] = df_first_iteration['filename'].apply(lambda fn: get_stats(fn))
df

# +
good_df = pd.DataFrame({
    'filename': pd.Series(glob.glob("/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/**/**/1/pc_align/*/*.json"))
})
good_df_first_iteration = pd.DataFrame({
    'filename': pd.Series(glob.glob("/data2/elilouis/generate_ee_dems_baker/mixed_timesift/individual_clouds/**/**/0/pc_align/*/*.json"))
})
good_df['run'] = 'nagap and ee images'

good_df['date'] = good_df['filename'].apply(lambda fn: fn.split('/')[6])
good_df['cluster'] = good_df['filename'].apply(lambda fn: fn.split('/')[7])
good_df['iteration'] = good_df['filename'].apply(lambda fn: fn.split('/')[8])
good_df['stats_first_iteration'] = good_df_first_iteration['filename'].apply(lambda fn: get_stats(fn))
good_df['stats'] = good_df['filename'].apply(lambda fn: get_stats(fn))
good_df
# -

df = pd.concat([df, good_df])

df['year'] = df['date'].apply(lambda x: x[:2])
df['Year-cluster'] = df['year'] + '-' + df['cluster']

alt.Chart(df).mark_tick(thickness=5).encode(
    alt.X("year:Q", scale=alt.Scale(zero=False)),
    alt.Y("run:N"),
    alt.Color("stats:Q", scale=alt.Scale(scheme="viridis"))
).properties(height=100, width=1000)

alt.Chart(df).mark_bar(thickness=5).encode(
    alt.X("run:N", scale=alt.Scale(zero=False), axis=alt.Axis(labels=False), title=None),
    alt.Y("stats:Q", scale=alt.Scale(scheme="viridis"), title='NMAD'),
    alt.Color("run:N")
).facet(
    'Year-cluster:N',
    columns=14
).properties(
    title={
      "text": ["Comparing NMAD of HSFM Pipeline runs when EE images are included"], 
      "subtitle": ["Mt Baker DEMs"],
    }
)

alt.Chart(df).mark_tick(thickness=5).encode(
    alt.X("stats:Q"),
    alt.Y("run:N")
#     alt.Facet('run', columns=1)
#     alt.Y("stats:Q"),
).properties(height=100, width=1000)

alt.Chart(df).mark_bar().encode(
    alt.X("stats:Q", bin=True),
    y='count()',
    color='run',
    facet='run'
)

df['run'].unique()

src = df[df['run'] == 'nagap images']
src['First Iteration'] = src['stats_first_iteration']
src['Second Iteration'] = src['stats']
src = src[src['First Iteration'] <= 3]
alt.Chart(src).transform_fold(
    ['First Iteration', 'Second Iteration'],
    as_ = ('key', 'value')
).mark_bar().encode(
    alt.X("key:N", title=None),
    alt.Y("value:Q"),
    alt.Color("key:N", title=None)
).facet(
    'Year-cluster:O',
    columns=14
).properties(
    title={
      "text": ["Comparing NMAD of 1st and 2nd iterations of the HSFM Pipeline"], 
      "subtitle": ["Mt Baker DEMs using only NAGAP Images"],
    }
)

src = df[df['run'] == 'nagap and ee images']
src['First Iteration'] = src['stats_first_iteration']
src['Second Iteration'] = src['stats']
src = src[src['First Iteration'] <= 3]
alt.Chart(src).transform_fold(
    ['First Iteration', 'Second Iteration'],
    as_ = ('key', 'value')
).mark_bar().encode(
    alt.X("key:N", title=None),
    alt.Y("value:Q"),
    alt.Color("key:N", title=None)
).facet(
    'Year-cluster:O',
    columns=14
).properties(
    title={
      "text": ["Comparing NMAD of 1st and 2nd iterations of the HSFM Pipeline"], 
      "subtitle": ["Mt Baker DEMs using NAGAP and EE Images"],
    }
)


