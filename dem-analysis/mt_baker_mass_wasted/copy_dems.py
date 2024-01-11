# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3.9.2 ('xdem')
#     language: python
#     name: python3
# ---

import glob
import os
from pathlib import Path
import shutil
import json 

if __name__ == "__main__":   
    # # Inputs
    #
    # * Inputs are written in a JSON.
    # * The inputs file is specified by the `HSFM_GEOMORPH_INPUT_FILE` env var
    # * One input may be overriden with an additional env var - `RUN_LARGER_AREA`. If this env var is set to "yes" or "no" (exactly that string, it will be used. If the env var is not set, the params file is used to fill in this variable. If some other string is set, a failure is thrown).

    # If you use the arg, you must run from CLI like this
    #
    # ```
    # HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute dem-analysis/mt_baker_mass_wasted/copy_dems.ipynb
    # ```

    BASE_PATH = os.environ.get("HSFM_GEOMORPH_DATA_PATH")
    print(f"retrieved base path: {BASE_PATH}")

    # Or set an env arg:
    if os.environ.get('HSFM_GEOMORPH_INPUT_FILE'):
        json_file_path = os.environ['HSFM_GEOMORPH_INPUT_FILE']
    else:
        json_file_path = 'inputs/mazama_inputs.json'

    with open(json_file_path, 'r') as j:
        params = json.loads(j.read())

    # +
    original_dems_path = os.path.join(BASE_PATH, params["copy"]["original_dems_path"])

    new_dems_path = os.path.join(BASE_PATH, params["inputs"]["dems_path"])
    # -

    # ls -lah {original_dems_path} | grep ".tif"

    # +
    dem_fn_list = glob.glob(os.path.join(original_dems_path, "*.tif"))
    dem_fn_list = sorted(dem_fn_list)

    for f in dem_fn_list:
        new_f = os.path.join(new_dems_path, Path(f).name)
        os.makedirs(Path(new_f).parent, exist_ok=True)
        shutil.copy(f, new_f)
