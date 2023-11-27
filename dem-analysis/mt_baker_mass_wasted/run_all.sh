##

## Run copy_dem_notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/copy_dems.ipynb --output outputs/copy_dems_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/copy_dems.ipynb --output outputs/copy_dems_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/copy_dems.ipynb --output outputs/copy_dems_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/copy_dems.ipynb --output outputs/copy_dems_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/copy_dems.ipynb --output outputs/copy_dems_rainbow.html && \
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/copy_dems_whole_mountain.ipynb --output outputs/copy_dems_whole_mountain.html

## Run uncertainty notebooks and xdem notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_rainbow.html && \

HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_coleman_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_deming_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_easton_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_mazama_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_rainbow_filtering_simple.html && \

## Run uncertainty notebooks and xdem notebooks for LARGER AREA
    ## MUST SWITCH TO LARGER_AREA = TRUE FOR THIS TO WORK!!
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_coleman_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_deming_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_easton_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_mazama_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_rainbow_largerarea.html && \

HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_deming_largerarea_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_coleman_largerarea_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_easton_largerarea_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_mazama_largerarea_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_rainbow_largerarea_filtering_simple.html && \

## Run Xsections notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_rainbow.html && \

# Run Transects notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_rainbow.html && \

# Run xdem_plot NB
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_plot.ipynb  --output outputs/xdem_plot.html

# Newest "extra" NBs
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/dem_dataset_info.ipynb  --output outputs/dem_dataset_info.html && \
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty_whole_mountain_combined.ipynb --output outputs/uncertainty_whole_mountain_combined.html
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_whole_mountain_combined.ipynb  --output outputs/xdem_whole_mountain_combined.html && \
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/slope_drainagearea_xdem.ipynb  --output outputs/slope_drainagearea_xdem.html

# Xdem Process NBs
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_rainbow.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' RUN_LARGER_AREA='no' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_easton.html

# Xdem Process NBs LARGER AREA
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_coleman_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_deming_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_mazama_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_rainbow_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' RUN_LARGER_AREA='yes' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_processes.ipynb --output outputs/xdem_processes_easton_largerarea.html

# Old NBs
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty_whole_mountain.ipynb --output outputs/uncertainty_whole_mountain.html
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_whole_mountain.ipynb  --output outputs/xdem_whole_mountain.html && \
jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem_whole_mountain_1947.ipynb  --output outputs/xdem_whole_mountain_1947.html && \
