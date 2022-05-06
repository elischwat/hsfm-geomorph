from hsfm.utils import dem_align_custom
import os
import glob

reference_dem = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif'
files = glob.glob("/data2/elilouis/earthdem/**/*dem.tif", recursive=True)
print(len(files))

for f in files:
    print(f'Aligning {f} to')
    print(f'Reference DEM {reference_dem}')
    output_path = f.replace('_dem.tif', '')
    print(f'Saving to {output_path}')

    dem_align_custom(
        reference_dem,
        f,
        output_directory=output_path
    )