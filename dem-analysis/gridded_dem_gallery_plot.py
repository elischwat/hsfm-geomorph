import pandas as pd
import geopandas as gpd
import rioxarray as rix
from pprint import pprint
from xrspatial import hillshade
import matplotlib.pyplot as plt
import glob
import os
import xarray as xr
from pathlib import Path
import json
import dask
dask.config.set({"array.slicing.split_large_chunks": False}) 

dataset_base_path = '/data2/elilouis/timesift/baker-ee-many/mixed_timesift_manual_selection/individual_clouds/'


# # Create low res version of all DEM files

# ## Filtered/Combined DEMs

# +
all_dem_files = glob.glob(os.path.join(dataset_base_path, '*', 'dem.tif'), recursive=True)
# pprint(all_dem_files)

# for f in all_dem_files:
#     print(f'Processing {f}')
#     new_f = f.replace('dem.tif', 'dem_lowres.tif')
#     print(f'Saving to {new_f}')
#     raster = rix.open_rasterio(f)
#     raster.rio.reproject(
#         raster.rio.crs,
#         (30,30)
#     ).rio.to_raster(
#        new_f
#     )
# -

# ## All raw DEMs

# +
all_raw_dem_files = glob.glob(os.path.join(dataset_base_path, '**', "*align.tif"), recursive=True)
# pprint(all_raw_dem_files)

# for f in all_raw_dem_files:
#     print(f'Processing {f}')
#     new_f = f.replace('align.tif', 'align_lowres.tif')
#     print(f'Saving to {new_f}')
#     raster = rix.open_rasterio(f)
#     raster.rio.reproject(
#         raster.rio.crs,
#         (30,30)
#     ).rio.to_raster(
#        new_f
#     )
# -

# # Create xr.Datasets of DEMs

# +
def create_dataset(individual_clouds_path, tif_name):
    """
    Depends on files being organized as is the output of the TimesiftPipeline classs.
    individual_clouds_path (str): Path ending with "individual_clouds"
    tif_name (str): type of file to use. options include 'dem', 'dod', and 'orthomosaic'
    """
    artificial_path = os.path.join(individual_clouds_path, '*', tif_name + '.tif')
    all_files = glob.glob(artificial_path, recursive=True)
    d = {}
    for f in all_files:
        tif_name = os.path.split(os.path.split(f)[0])[-1]
        print(f'Reading {f}')
        d[tif_name] = rix.open_rasterio(
            f, 
            masked=True
        )
    return d


def create_unfiltered_dataset(individual_clouds_path):
    """
    Depends on files being organized as is the output of the TimesiftPipeline classs.
    individual_clouds_path (str): Path ending with "individual_clouds"
    tif_name (str): type of file to use. options include 'dem', 'dod', and 'orthomosaic'
    """
    stats_artificial_path = os.path.join(individual_clouds_path, '**', "spoint2point_bareground-trans_source-DEM_dem_align", "*align_stats.json")
    dem_artificial_path = os.path.join(individual_clouds_path, '**', "spoint2point_bareground-trans_source-DEM_dem_align", "*align_lowres.tif")
    all_stats_files = glob.glob(stats_artificial_path, recursive=True)
    all_dem_files = glob.glob(dem_artificial_path, recursive=True)    
    d = {}    
    for dem_f, stats_f in zip(all_dem_files, all_stats_files):
        assert (
            stats_f.split('individual_clouds')[-1].split('/')[1:3] == dem_f.split('individual_clouds')[-1].split('/')[1:3] 
        )
        tif_name = '/'.join(
            dem_f.split('individual_clouds')[-1].split('/')[1:3]
        )
        print(f'Processing {dem_f}')
        print(tif_name)
        with open(stats_f) as s:
            stats = json.load(s)
        d[tif_name] = rix.open_rasterio(
            dem_f, 
            masked=True
        )
        d[tif_name].attrs['nmad'] =  stats['after_filt']['nmad']
    return d

# for dem_f, stats_f in zip(all_raw_dem_files, all_raw_stats_files):
#     assert (
#         dem_f.split('individual_clouds')[-1].split('/')[0] == stats_f.split('individual_clouds')[-1].split('/')[0]
#     )
#     print(f'Processing {dem_f}')
#     new_dem_f = dem_f.replace('align.tif', 'align_lowres.tif')
#     print(f'Saving to {new_dem_f}')
#     with open(stats_f) as s:
#         d = json.load(s)



# -

dem_lowres_dataset = create_dataset(dataset_base_path, 'dem_lowres')

dem_unfilt_lowres_dataset = create_unfiltered_dataset(dataset_base_path)

dem_lowres_dataset.keys()

dem_unfilt_lowres_dataset.keys()

# ### Create date-merged version of raw DEMs

# +
merged_dem_unfilt_lowres_dataset = {}
for date in set([
    k.split('/')[0] for k in dem_unfilt_lowres_dataset.keys()
]):
    print(f'Merging {date}')
    for k in dem_unfilt_lowres_dataset.keys():
        if date in k:
            if merged_dem_unfilt_lowres_dataset.get(date): #list already created
                merged_dem_unfilt_lowres_dataset[date] = merged_dem_unfilt_lowres_dataset[date] + [dem_unfilt_lowres_dataset[k]]
            else:
                merged_dem_unfilt_lowres_dataset[date] = [dem_unfilt_lowres_dataset[k]]
                
from rioxarray.merge import merge_arrays
for date, dems_to_merge in merged_dem_unfilt_lowres_dataset.items():
    merged_dem_unfilt_lowres_dataset[date] = merge_arrays(dems_to_merge)
# -

dem_lowres_dataset.keys()

dem_unfilt_lowres_dataset.keys()

merged_dem_unfilt_lowres_dataset.keys()

# # Plot

# ## All DEMs by cluster

len(dem_unfilt_lowres_dataset.keys())

# +
rows = 5
columns = 9

fig = plt.figure(figsize=(25,14))

bounds = dem_unfilt_lowres_dataset['92_9.0_18.0/cluster0'].rio.bounds()

for i, (key, raster) in enumerate(dem_unfilt_lowres_dataset.items()):
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
plt.savefig('dem_gallery_all_dems_by_cluster.png')
plt.show()
# -

# ## All DEMS by Date

# +
rows = 3
columns = 6

fig = plt.figure(figsize=(25,14))

bounds = merged_dem_unfilt_lowres_dataset['92_9.0_18.0'].rio.bounds()

for i, (key, raster) in enumerate(sorted(merged_dem_unfilt_lowres_dataset.items())):
    raster_clipped = raster.rio.clip_box(*bounds)
    raster_clipped = raster_clipped.rio.pad_box(*bounds)
    
    ax = plt.subplot(rows, columns, i + 1, aspect='auto')
    ax.imshow(raster_clipped.squeeze(), interpolation='none')
    ax.imshow(hillshade(raster_clipped.squeeze()), interpolation='none', cmap='gray_r', alpha=0.5)
    
    # ax.set_xlim(bounds[0],bounds[2])
    # ax.set_ylim(bounds[1],bounds[3])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(key.replace('.0',''), size=18)
    ax.set_facecolor('black')

fig.suptitle('Mount Baker DEM Gallery', fontsize=18)
fig.set_facecolor("w")
plt.tight_layout()
plt.savefig('dem_gallery_all_dems_by_date.png')
plt.show()
# -

# ## Good DEMs by Date

# +
rows = 3
columns = 6

fig = plt.figure(figsize=(15,7))

bounds = dem_lowres_dataset['77_9.0_27.0'].rio.bounds()

for i, (key, raster) in enumerate(dem_lowres_dataset.items()):
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

# ## Good NAGAP DEMs by Date

# +
nagap_dems_only_dataset = dem_lowres_dataset.copy()
nagap_dems_only_dataset.pop('47_9.0_14.0')
nagap_dems_only_dataset.pop('50_9.0_2.0')
nagap_dems_only_dataset.pop('79_9.0_14.0')

rows = 4
columns = 4

fig = plt.figure(figsize=(10,15))

bounds = nagap_dems_only_dataset['77_9.0_27.0'].rio.bounds()

for i, (key, raster) in enumerate(nagap_dems_only_dataset.items()):
    raster_clipped = raster.rio.clip_box(*bounds)
    raster_clipped = raster_clipped.rio.pad_box(*bounds)
    
    ax = plt.subplot(rows, columns, i + 1, aspect='auto')
    ax.imshow(raster_clipped.squeeze(), interpolation='none')
    ax.imshow(hillshade(raster_clipped.squeeze()), interpolation='none', cmap='gray_r', alpha=0.5)
    
    # ax.set_xlim(bounds[0],bounds[2])
    # ax.set_ylim(bounds[1],bounds[3])
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(key,size=18)
    ax.set_facecolor('black')

fig.suptitle('Mount Baker DEM Gallery (NAGAP images only)', fontsize=18)
fig.set_facecolor("w")
plt.tight_layout()
plt.savefig('dem_gallery_filtered_nagap_dems_by_date.png')
plt.show()
# -

# ## Friedrich's original plotting code:

# +
# fig = plt.figure(figsize=(10,15))

# for i in range(rows*columns):
#     try:
#         ds = rasterio.open(input_set[ia],masked=True)
#         ax = plt.subplot(rows, columns, i + 1, aspect='auto')

#         show(ds,ax=ax,interpolation='none',
#              cmap=cmap,
#              vmin = vmin, vmax = vmax)
        
#         ds = rasterio.open(hillshades[i],masked=True)
#         show(ds,ax=ax,interpolation='none',cmap='gray',alpha=0.5)
  
#         ax.set_xlim(bounds[0],bounds[2])
#         ax.set_ylim(bounds[1],bounds[3])
#         ax.set_xticks(())
#         ax.set_yticks(())
#         ax.set_title(mosaics_dates[i],size=18)
#         ax.set_facecolor('black')
# #         ctx.add_basemap(ax, 
# #                         source="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
# #                         crs=ds.crs.to_string(), alpha = 0.7, zorder=-1)
#     except:
#         pass

# ax = fig.add_axes([0.5, 0.05, 0.33, 0.07])
# ax.set_xlim(bounds[0],bounds[2])
# ax.set_ylim(bounds[1],bounds[3])
# ax.axis('off')
# sb = ScaleBar(1, 
#               location='upper center',
#               length_fraction=0.5,height_fraction=0.1)
# pos1 = ax.get_position()
# pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height] 
# ax.set_position(pos2)
# ax.add_artist(sb)

# cbar_ax = fig.add_axes([0.5, 0.05, 0.33, 0.15])
# cbar_ax.axis('off')
# sm = plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin = -spread, vmax = spread))
# cbar = fig.colorbar(sm, ax=cbar_ax, fraction=1, pad=0.5,aspect=20, 
#                     orientation='horizontal')
# cbar.set_label(label='Elevation difference (m)', size=12)



# fig.suptitle('Mount Baker elevation difference gallery', fontsize=18,y=0.99)
# plt.tight_layout()



# out = os.path.join(final_plots_dir,'dems_baker.png')
# plt.savefig(out, dpi=300, bbox_inches='tight')
# -

# # Count original images and those used

# dataset_base_path = '/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds_2_4/'
dataset_base_path = '/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/'

nagap_images_avail_df = pd.read_csv('/data2/elilouis/timesift/baker-ee-many/metashape_metadata.csv')

final_processing_unaligned_metadata_files = glob.glob(
    os.path.join(dataset_base_path, '*', 'cluster[0-9]', '0', 'metaflow_bundle_adj_unaligned_metadata.csv')
)
cluster_processing_unaligned_metadata_files = glob.glob(
    os.path.join(dataset_base_path, '*', 'single_date_multi_cluster_bundle_adjusted_unaligned_metashape_metadata*')
)

final_processing_unaligned_cams_df = pd.concat(
    (pd.read_csv(f) for f in final_processing_unaligned_metadata_files),
    ignore_index=True
)

cluster_processing_unaligned_cams_df = pd.concat(
    (pd.read_csv(f) for f in cluster_processing_unaligned_metadata_files),
    ignore_index=True
)

print(len(final_processing_unaligned_cams_df), len(cluster_processing_unaligned_cams_df))

print(len(nagap_images_avail_df))

cluster_processing_unaligned_cams_df[
    cluster_processing_unaligned_cams_df['image_file_name'].str.contains('92')
]

cluster_processing_unaligned_cams_df[
    cluster_processing_unaligned_cams_df['image_file_name'].str.contains('92')
]


