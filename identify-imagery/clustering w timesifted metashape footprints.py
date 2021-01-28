# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import hsfm
import contextily as ctx
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from shapely.ops import cascaded_union

metadata_df_file = "/home/elilouis/hipp/examples/fiducial_proxy_detection/input_data/nagap_image_metadata.csv"
metadata_df = pd.read_csv(metadata_df_file, index_col=0)

project_file = "/data2/elilouis/test.geojson"
gdf = gpd.read_file(project_file)
# project_file = "/data2/elilouis/baker/multi_epoch_cloud/multi_epoch_densecloud.psx"
# gdf = hsfm.metashape.image_footprints_from_project(project_file, points_per_side=25)

metadata_df.head(2)

gdf.head(2)

joined = gpd.GeoDataFrame(pd.merge(
    metadata_df[['fileName', 'Roll', 'Year', 'Month', 'Day']],
    gdf,
    left_on='fileName',
    right_on='filename'
))

joined = joined.to_crs(epsg=3857)



# +
def intersects_with_threshold(left, right, overlap_fraction_threshold = 0.1):
    if (left.intersects(right)):
        intersection = left.intersection(right)
        if intersection.area/left.area >= overlap_fraction_threshold and \
            intersection.area/right.area >= overlap_fraction_threshold:
            return True
        else:
            return False
    else:
        return False
def polygon_overlaps_with_any(polygon, polygon_list, overlap_percent_threshold = 0.1):
    return any([
        intersects_with_threshold(
            polygon, other[1], 
            overlap_percent_threshold
        ) for other in polygon_list
    ])
def redistribute_small_groups(list_of_lists_of_polygons, min_group_size = 3):
    new_list_of_lists = []
    polygons_to_reassign = []
    #iterate over all lists of polygons, saving the large-enough lists/clusters
    # and putting the polygons in too-small lists/clusters in a separate list
    for list_of_polygons in list_of_lists_of_polygons:
        if(len(list_of_polygons) >= min_group_size):
            new_list_of_lists.append(list_of_polygons)
        else:
            polygons_to_reassign = polygons_to_reassign + list_of_polygons
    # for each polygon that needs reassigning,
    # iterate over all lists/clusters and see with which cluster the polygon has
    # the greatest overlap
    for filename, polygon in polygons_to_reassign:
        max_overlap = 0
        list_index_for_reassignment = None
        for i, polygon_list in enumerate(new_list_of_lists):
            cluster_union_polygon = cascaded_union([poly for filename,poly in  polygon_list])
            intersection_polygon = cluster_union_polygon.intersection(polygon)
            overlap_percent = intersection_polygon.area/polygon.area
            if overlap_percent > max_overlap:
                max_overlap = overlap_percent
                list_index_for_reassignment = i
        if list_index_for_reassignment:
            new_list_of_lists[i] = new_list_of_lists[i].append((filename, polygon))
        else:
            print(f"Could not find cluster for image {filename}...dropping image.")
    return new_list_of_lists

def group_by_polygon_overlap(gdf, overlap_percent_threshold=0.1):
    list_of_groups = []
    #instantiate the list of groups with the first filename-geometry pair
    list_of_groups.append([(gdf.filename.iloc[0],gdf.geometry.iloc[0])])
    #for each geometry, see if it overlaps with any 
    for k, row in gdf.iloc[1:].iterrows():
        geom = row.geometry
        filename = row.filename
        has_been_grouped = False
        for group in list_of_groups:
            if polygon_overlaps_with_any(geom, group, overlap_percent_threshold):
                group.append((filename,geom))
                has_been_grouped = True
        if has_been_grouped == False:
            list_of_groups.append([(filename,geom)])
            
    # if a group has less than three images, add those images to other groups
    # add image to the group with the greatest amount of overlap
#     list_of_groups = redistribute_small_groups(list_of_groups)
    
    gdf_list = []
    for i, cluster in zip(range(0,len(list_of_groups)), list_of_groups):
        gdf = gpd.GeoDataFrame(cluster)
        gdf['cluster'] = i
        gdf_list.append(gdf)
    df = pd.concat(gdf_list)
    df = df.rename({
        0:'filename',
        1:'geometry',
    }, axis = 1)
    return gpd.GeoDataFrame(df)


# -

cluster_to_df_dict = {}
for (roll, year, month, day), gdf in joined.groupby(['Roll', 'Year', 'Month', 'Day']):
    print(f'Original length: {len(gdf)}')
    gdf_clustered = group_by_polygon_overlap(gdf, 0.10)
    print(f'Clustered length: {len(gdf_clustered)}')
    print(f'Number of clusters: {len(gdf_clustered.cluster.unique())}')
    ax = gdf_clustered.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', column='cluster')
    ctx.add_basemap(ax)
    plt.title(f'NAGAP Images on {day}/{month}/{year}')
    plt.show()


