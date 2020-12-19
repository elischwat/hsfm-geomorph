# hsfm-geomorph
Using HSFM DEMs to understand decadal-scale geomorphology in the PNW.

1. identify-imagery\t\t Examing NAGAP (and other?) image datasets and creating subsets for DEM processing
2. create_dems\t\t Create DEMs using https://github.com/friedrichknuth/hsfm/
3. ...

All python scripts/notebooks in the repository depend on a large and locally hosted dataset.

Note the `data_dir` variable set at the beginning of all scripts/notebooks. You can set in manually or rely on an environmental variable `data_dir`.

I can use the `download-data.sh` script but you cannot. Future improvements will have the dataset publically available.