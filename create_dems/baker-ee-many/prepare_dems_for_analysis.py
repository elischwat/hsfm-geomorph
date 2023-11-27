#%%
import geopandas as gpd
import rioxarray as rix
from rioxarray import merge
import glob
import os
from shapely.geometry import box
import shutil
import hsfm.utils

"""
NOTE: Before running this, i placed this file
/data2/elilouis/timesift/baker-ee-many/mixed_timesift_manual_selection/individual_clouds/47_9.0_14.0/cluster0/0/pc_align/point2plane-trans_source-DEM_dem_align/point2plane-trans_source-DEM_2015_nuth_x-0.05_y-0.44_z+0.01_align.tif
 in the directory
/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/final_products/dems/"
manually, and renamed it "47_9_14_cluster0.tif".
I do this because this 1947 product is better in the other timesift run. All other products I use from this run.
"""
#%%
dems_path = "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/final_products/dems/"

# Identify watersheds that will be used to crop DEMs
wshed_names  = [
    "Wells Creek",
    "Glacier Creek",
    "Upper Middle Fork Nooksack River",
    "Lake Shannon-Baker River",
    "Swift Creek",
    "Lower Baker Lake"
]

# use original watersheds:
# wsheds_gdf = gpd.read_file(
#     "/data2/elilouis/hsfm-geomorph/data/NHDPLUS_H_1711_HU4_GDB/NHDPLUS_H_1711_HU4_GDB.gdb",
#     layer=66
# ).to_crs('EPSG:32610')
# use watersheds modified to include static area near watershed boundaries.
#   NOT the actual watersheds
wsheds_gdf = gpd.read_file("/data2/elilouis/hsfm-geomorph/data/mt_baker_mass_wasted/watersheds.geojson")
wsheds_gdf = wsheds_gdf[wsheds_gdf['Name'].isin(wshed_names)]


# %%
## Mosaic DEMs so we have one DEM per date and save them to file
all_dem_files = os.listdir(dems_path)
dates = list(set([d.split("_cluster")[0] for d in all_dem_files]))
mosaiced_dems_path = os.path.join(dems_path.replace("dems", "dems_mosaiced"))
date_to_mosaic_dem_dict = {}
for date in dates:
    filtered_dem_files = [f for f in all_dem_files if date in f]
    print(date)
    print(filtered_dem_files)
    print()
    new_mosaiced_location = os.path.join(mosaiced_dems_path, date + '.tif')
    if len(filtered_dem_files) == 1:
        # just copy the single cluster file into the new directory, and open it up
        shutil.copyfile(os.path.join(dems_path, filtered_dem_files[0]), new_mosaiced_location)
        date_to_mosaic_dem_dict[date] = rix.open_rasterio(new_mosaiced_location)
    else:
        # mosaic all the files into one DEM using ASPs dem_mosaic
        hsfm.asp.dem_mosaic(
            new_mosaiced_location, 
            [os.path.join(dems_path, f) for f in filtered_dem_files]
        )
        date_to_mosaic_dem_dict[date] = rix.open_rasterio(new_mosaiced_location)
print("Generated the following mosaiced dems:")
all_mosaiced_dems = glob.glob(os.path.join(mosaiced_dems_path, "*.tif"))
print(all_mosaiced_dems)


# %%
# For each mosaiced DEM, clip by watershed boundary and save
watershed_dems_path = os.path.join(dems_path.replace("dems", "dems_by_watershed_and_date"))
for f in all_mosaiced_dems:
    print(f"Splitting {f}")
    date_fn = os.path.join(mosaiced_dems_path, f)
    date = f.replace('.tif', '')
    raster = rix.open_rasterio(date_fn)
    for idx, wshed_row in wsheds_gdf.iterrows():
        wshed_geom = wshed_row['geometry']
        if box(*raster.rio.bounds()).intersects(wshed_geom):
            clipped = raster.rio.clip([wshed_geom])
            parent_path = os.path.join(
                watershed_dems_path, 
                wshed_row['Name'].replace(' ', '_').replace('-', '_').lower(),
            )
            os.makedirs(parent_path, exist_ok=True)
            clipped.rio.to_raster(
                os.path.join(parent_path, date_fn.split('/')[-1])
            )
# %%
# EARTHDEM TIME - For each EarthDEM, mask by the associated bitmask, copy the result into the `dems_mosaiced` folder, then clip by watershed boundary and save

dem2013_fn = "/data2/elilouis/earthdem/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1_dem_dem_align/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1_dem_2015_nuth_x+0.60_y-0.22_z+0.55_align_ct_at_corrected.tif"
bitmask_2013_fn = "/data2/elilouis/earthdem/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1_bitmask.tif"
# masked file will be put here
new_dem2013_fn = "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/final_products/dems_mosaiced/2013_09_13.tif" 

dem2019_fn = "/data2/elilouis/earthdem/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1_dem_dem_align/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1_dem_2015_nuth_x+4.50_y+3.69_z-1.50_align_ct_at_corrected.tif"
bitmask_2019_fn = "/data2/elilouis/earthdem/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1_bitmask.tif"
# masked file will be put here
new_dem2019_fn = "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/final_products/dems_mosaiced/2019_10_11.tif"

dem2013 = rix.open_rasterio(dem2013_fn, chunks=True).squeeze()
bitmask2013 = rix.open_rasterio(bitmask_2013_fn, chunks=True).squeeze()

demvalues = dem2013.values
demvalues[bitmask2013.values != 0] = dem2013._FillValue
dem2013.values = demvalues

dem2013.rio.to_raster(new_dem2013_fn)

dem2019 = rix.open_rasterio(dem2019_fn, chunks=True).squeeze()
bitmask2019 = rix.open_rasterio(bitmask_2019_fn, chunks=True).squeeze()

demvalues = dem2019.values
demvalues[bitmask2019.values != 0] = dem2019._FillValue
dem2019.values = demvalues

dem2019.rio.to_raster(new_dem2019_fn)

dem2013_fn = "/data2/elilouis/earthdem/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1_dem_dem_align/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1_dem_2015_nuth_x+0.60_y-0.22_z+0.55_align_ct_at_corrected_masked.tif"

dem2019_fn = "/data2/elilouis/earthdem/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1_dem_dem_align/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1_dem_2015_nuth_x+4.50_y+3.69_z-1.50_align_ct_at_corrected_masked.tif"

for idx, wshed_row in wsheds_gdf.iterrows():
    for date_fn, dem_fn in [
        ('2013_09_13.tif', dem2013_fn),
        ('2019_10_11.tif', dem2019_fn)
    ]:
        wshed_geom = wshed_row['geometry']
        raster = rix.open_rasterio(dem_fn)
        clipped = raster.rio.clip([wshed_geom])
        parent_path = os.path.join(
            watershed_dems_path, 
            wshed_row['Name'].replace(' ', '_').replace('-', '_').lower(),
        )
        clipped.rio.to_raster(
            os.path.join(parent_path, date_fn)
        )

# %%
# REFERENCE DEM TIME - copy into the `dems_mosaiced` folder, then clip by watershed boundary and save
ref_dem_fn = "/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif"
new_ref_dem_fn = "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds/final_products/dems_mosaiced/2015_09_01.tif"

shutil.copy(ref_dem_fn, new_ref_dem_fn)
raster = rix.open_rasterio(ref_dem_fn)
for idx, wshed_row in wsheds_gdf.iterrows():
    wshed_geom = wshed_row['geometry']
    clipped = raster.rio.clip([wshed_geom])
    parent_path = os.path.join(
        watershed_dems_path, 
        wshed_row['Name'].replace(' ', '_').replace('-', '_').lower(),
    )
    clipped.rio.to_raster(
        os.path.join(parent_path, "2015.tif")
    )


# Now manually go through all the date-specific files in each watershed and remove the ones that don't have good enough coverage to actually use.

# I DONT THINK WE SHOULD DO THIS ALIGNMENT STEP!!!!
# # For each set of DEMs by watershed, run dem_align.py with 2015 reference DEM as reference
# # Make sure to do the correct order BTW (so new minus old and the *align.png files are visually correct.).
# #%%
# watershed_dirs = os.listdir(watershed_dems_path)
# REF_DEM_NAME = "2015.tif"
# for d in watershed_dirs:
#     full_dir_path = os.path.join(watershed_dems_path, d)
#     for dem_file in os.listdir(full_dir_path):
#         if dem_file != REF_DEM_NAME:
#             dem_to_align_path = os.path.join(full_dir_path, dem_file)
#             ref_dem_path = os.path.join(full_dir_path, REF_DEM_NAME)
#             hsfm.utils.dem_align_custom(
#                 ref_dem_path, 
#                 dem_to_align_path, 
#                 None
#             )

# # Copy the results of dem_align.py into a combined directory.
# # %%
# final_dems_path = watershed_dems_path.replace("dems_by_watershed_and_date", "dems_by_watershed_and_date_aligned")
# all_aligned_dems = glob.glob(
#     os.path.join(watershed_dems_path, "**/*align.tif"),
#     recursive=True
# )
# for f in all_aligned_dems:
#     split = f.split("_dem_align")[0]
#     copied_aligned_dem_fn = split.replace("dems_by_watershed_and_date", "dems_by_watershed_and_date_aligned") + '.tif'
#     os.makedirs(os.path.dirname(copied_aligned_dem_fn), exist_ok=True)
#     shutil.copyfile(f, copied_aligned_dem_fn)

# # %%
# #and bring the 2015 DEMs into that directory too
# by_watershed_2015_dems = glob.glob(os.path.join(watershed_dems_path, "**/2015.tif"), recursive=True)
# for f in by_watershed_2015_dems:
#     shutil.copyfile(
#         f, 
#         f.replace("dems_by_watershed_and_date", "dems_by_watershed_and_date_aligned")
#     )

# %%
# final_dem_fns = glob.glob(os.path.join(final_dems_path, "**/*.tif"), recursive=True)

# %%
final_dem_fns = glob.glob(os.path.join(watershed_dems_path, "**/*.tif"))


# %%
# Rename DEMs so they all have the file name formatted YYYY_MM_DD - do this for the mosaiced DEMs and the mosaiced DEMs
from pathlib import Path
for list_of_dems in [all_mosaiced_dems, final_dem_fns]:
    for f in list_of_dems:
        stem = Path(f).stem
        pieces = stem.split("_")
        # print(f"Fixing {stem} in pieces {pieces}")
        if len(pieces) == 1:
            if len(pieces[0]) == 4:
                new_stem = '_'.join([pieces[0], '09', '01']) #default date of September 1 for those missing dates
            else:
                raise ValueError(f"I don't know what to do with the filename {stem}")
        elif len(pieces) == 3:
            new_pieces = []
            # Handle case where year is only 2 digits - we assume its in the 1900s and prepend '19'
            if len(pieces[0]) == 2:
                new_pieces.append('19' + pieces[0])
            else:
                new_pieces.append(pieces[0])
            # Handle case where month is only one digit - we assume its a single digit month and prepend '0'
            if len(pieces[1]) == 1:
                new_pieces.append('0' + pieces[1])
            else:
                new_pieces.append(pieces[1])
            # Handle case where day is only one digit - we assume its a single digit day and prepend '0'
            if len(pieces[2]) == 1:
                new_pieces.append('0' + pieces[2])
            else:
                new_pieces.append(pieces[2])
            # print(f"I've collected pieces {new_pieces}")
            new_stem = '_'.join(new_pieces)
        print(f"Replacing {stem} with {new_stem}")
        os.rename(
            f,
            os.path.join(f.replace(stem, new_stem))
        )

    # %%
