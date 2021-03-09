# Provide path to directory with the downloaded zip files in it
base_path=/data2/elilouis/hsfm-geomorph/data/reference_dem_highres/hood/
FILES="${base_path}*.zip"

# Loop over zip files and unzip them
for f in $FILES
do
dirpath=${f%/*}
echo -e "Unzipping ${f} to ${dirpath}"
unzip $f -d $dirpath
done

# Translate all adf files
find . -name "*w001001.adf" -exec sh -c 'gdal_translate "$1" "${1%.adf}.tif"' _ {} \;