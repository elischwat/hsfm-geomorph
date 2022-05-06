##

## Run uncertainty notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/uncertainty.ipynb  --output outputs/uncertainty_rainbow.html

## Run XDEM notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_coleman_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_deming_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_easton_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_mazama_filtering_simple.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_rainbow_filtering_simple.html

## Run XDEM notebooks (larger area)
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_coleman_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_deming_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_easton_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_mazama_largerarea.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xdem.ipynb  --output outputs/xdem_rainbow_largerarea.html

## Run Xsections notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/xsections.ipynb  --output outputs/xsections_rainbow.html

# Run Transects notebooks
HSFM_GEOMORPH_INPUT_FILE='inputs/coleman_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_coleman.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/deming_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_deming.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/easton_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_easton.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/mazama_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_mazama.html && \
HSFM_GEOMORPH_INPUT_FILE='inputs/rainbow_inputs.json' jupyter nbconvert --execute --to html dem-analysis/mt_baker_mass_wasted/transects.ipynb  --output outputs/transects_rainbow.html

