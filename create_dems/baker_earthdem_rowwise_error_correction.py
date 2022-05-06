import hsfm
import xdem as du
import geoutils as gu
import numpy as np
import os
import matplotlib.pyplot as plt
from demcoreg import coreglib
import glob


base_paths = [
    "/data2/elilouis/earthdem/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1/SETSM_WV03_20191011_104001005314D800_1040010053D56500_2m_lsf_seg1_v1_dem_dem_align/",
    "/data2/elilouis/earthdem/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1/SETSM_WV01_20130913_1020010025E22B00_1020010023CCFE00_2m_lsf_seg1_v1_dem_dem_align/",
    "/data2/elilouis/earthdem/SETSM_WV01_20200905_102001009A8A1800_102001009C852B00_2m_lsf_seg3_v1/SETSM_WV01_20200905_102001009A8A1800_102001009C852B00_2m_lsf_seg3_v1_dem_dem_align/"
]

def at_ct_correction_dem_align_folder(dem_align_dir,min_axes_count=50):
       src_dem = sorted(glob.glob(os.path.join(dem_align_dir,'*align.tif')))[0]
       dh = sorted(glob.glob(os.path.join(dem_align_dir,'*align_diff.tif')))[0]
       dh_filt = sorted(glob.glob(os.path.join(dem_align_dir,'*align_diff_filt.tif')))[0]
       coreglib.ct_at_correction_wrapper(src_dem,dh,dh_filt,min_axes_count=50, ct_only=True)
   
for path in base_paths:
    at_ct_correction_dem_align_folder(path)