# %%
import os
import glob
import shutil

# %%
base_path = "/data2/elilouis/timesift/baker-ee-many/mixed_timesift/individual_clouds"

ortho_paths = glob.glob(os.path.join(base_path, "**", "*align_orthomosaic.tif"), recursive=True)
dem_paths = [f.replace("align_orthomosaic.tif", "align.tif") for f in ortho_paths]
dod_paths = [f.replace("align_orthomosaic.tif", "align_diff.tif") for f in ortho_paths]

# %%
for path_list in [ortho_paths, dem_paths, dod_paths]:
    assert all([
        os.path.exists(f) for f in path_list
    ])

# %%
final_products_path = os.path.join(base_path, "final_products")

# %%
for path in ortho_paths:
    name = path.split("/individual_clouds/")[1].split("/")[:2]
    name = '_'.join(name).replace(".0", "") + '.tif'
    new_path = os.path.join(final_products_path, 'orthomosaics', name)
    print(f"Copying {path} to {new_path}")
    dir_name, _ = os.path.split(new_path)
    os.makedirs(dir_name,exist_ok=True)
    shutil.copy(
        path, 
        new_path
    )

# %%
for path in dem_paths:
    name = path.split("/individual_clouds/")[1].split("/")[:2]
    name = '_'.join(name).replace(".0", "") + '.tif'
    new_path = os.path.join(final_products_path, 'dems', name)
    print(f"Copying {path} to {new_path}")
    dir_name, _ = os.path.split(new_path)
    os.makedirs(dir_name,exist_ok=True)
    shutil.copy(
        path, 
        new_path
    )

for path in dod_paths:
    name = path.split("/individual_clouds/")[1].split("/")[:2]
    name = '_'.join(name).replace(".0", "") + '.tif'
    new_path = os.path.join(final_products_path, 'dods', name)
    print(f"Copying {path} to {new_path}")
    dir_name, _ = os.path.split(new_path)
    os.makedirs(dir_name,exist_ok=True)
    shutil.copy(
        path, 
        new_path
    )

#%%
import rioxarray as rix
for file in os.listdir(os.path.join(final_products_path, 'orthomosaics')):
    print(
        rix.open_rasterio(os.path.join(final_products_path, 'orthomosaics', file)).rio.crs
    )
    print(
        rix.open_rasterio(os.path.join(final_products_path, 'orthomosaics', file)).rio.resolution()
    )

# %%
# Create low resolution ortho products
LOW_RES_FACTOR = 5

import subprocess
for file_name in os.listdir(os.path.join(final_products_path, 'orthomosaics')):    
    file_path = os.path.join(final_products_path, 'orthomosaics', file_name)
    resolution = rix.open_rasterio(file_path).rio.resolution()
    new_file_path = file_path.replace("/orthomosaics/", "/orthomosaics_lowres/")

    dir_name, _ = os.path.split(new_file_path)
    os.makedirs(dir_name,exist_ok=True)

    subprocess.call(['gdal_translate', '-of', 'GTiff', '-tr', str(resolution[0]*LOW_RES_FACTOR), str(resolution[1]*LOW_RES_FACTOR), '-co', 'TILED=YES', file_path, new_file_path])

    # %%
