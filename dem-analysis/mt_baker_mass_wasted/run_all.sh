


export HSFM_GEOMORPH_DATA_PATH='/storage/elilouis/'

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='yes' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='yes' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='yes' python uncertainty.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='yes' python uncertainty.py

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='yes' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='yes' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='yes' python xdem_create.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='yes' python xdem_create.py

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='yes' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='yes' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='yes' python xdem_processes.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='yes' python xdem_processes.py

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' python xdem_processes_bounding.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' python xdem_processes_bounding.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' python xdem_processes_bounding.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' python xdem_processes_bounding.py

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' python xsections.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' python xsections.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' python xsections.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' python xsections.py

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' python transects.py
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' python transects.py
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' python transects.py
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' python transects.py

python dem_dataset_info.py

python uncertainty_whole_mountain_combined.py

python xdem_create_whole_mountain_combined.py

python slope_drainagearea_xdem.py

python xdem_plot.py

python create_glacier_area_change_table.py

python power_law_relationships.py

python terrace_erosion_reanalysis.py

python create_supplemental_tables.py