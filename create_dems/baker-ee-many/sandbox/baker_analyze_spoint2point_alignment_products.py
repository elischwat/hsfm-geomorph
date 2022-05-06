# Analyze point2plane aligned products from recent baker-ee-many runs
import glob
from hsfm.utils import dem_align_custom



reference_dem = '/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/baker/raw_tifs/baker_2015/2015.tif'
all_spoint2point_dems = glob.glob("/data2/elilouis/timesift/baker-ee-many/mixed_timesift_manual_selection/individual_clouds/**/spoint2point-trans_source-DEM.tif", recursive=True)
print(all_spoint2point_dems)


for f in all_spoint2point_dems:
    print(f'Aligning {f} to')
    print(f'Reference DEM {reference_dem}')
    output_path = f.replace('.tif', '')
    print(f'Saving to {output_path}')

    dem_align_custom(
        reference_dem,
        f,
        output_directory=output_path
    )