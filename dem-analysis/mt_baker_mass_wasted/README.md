Mount Baker Mass Wastes - Analysis of land surface changes over Mount Baker.
---
Notebooks in the mt_baker_mass_wasted project must be run in a particular order.

Input files should live in the inputs/ directory.

Examples of how to run notebooks with specific input files in the run_all.sh file:
```
run_all.sh
```

Analysis notebooks to run in order:
---
```
copy_dems.ipynb
copy_dems_whole_mountain.ipynb

uncertainty.ipynb
uncertainty_whole_mountain.ipynb
uncertainty_whole_mountain_1947.ipynb

xdem.ipynb
xdem_whole_mountain.ipynb
xdem_whole_mountain_1947.ipynb

xsections.ipynb
transects.ipynb

xdem_plot.ipynb

create_uncertainty_table.ipynb

dem_dataset_info.ipynb

slope_drainagearea.ipynb
```

One-off notebooks:
---
```
create_thresholded_1947_dataset.ipynb
```