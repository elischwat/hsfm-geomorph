import numpy as np
import shapely
import geopandas as gpd
import rasterio
import pandas as pd
from shapely.geometry import LineString, Point

def get_raster_values_for_line(linestring, raster_file, band=1):
    """Assumes the CRS of the linestring is that of the raster_file."""
    raster = rasterio.open(raster_file)
    coords = list(zip(*linestring.coords.xy))
    values = [x[band - 1] for x in raster.sample(coords)]
    values = [x if x != raster.nodata else np.NaN for x in values]
    df = pd.DataFrame({
        'X': [x for x,y in coords],
        'Y': [y for x,y in coords],
        'raster_value': values
    })
    # add in the "path" distance
    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df.X, df.Y),
        crs=raster.crs
    )
    df['path_distance'] = pd.Series(gdf.distance(
        gpd.GeoDataFrame(gdf.shift(1), crs=gdf.crs)
    ).fillna(0)).cumsum()
    return df

def increase_line_complexity(linestring, n_points):
    """
    linestring (shapely.geometry.linestring.LineString): 
    n_points (int): target number of points
    """
    # or to get the distances closest to the desired one:
    # n = round(line.length / desired_distance_delta)
    distances = np.linspace(0, linestring.length, n_points)
    points = [linestring.interpolate(distance) for distance in distances]
    return shapely.geometry.linestring.LineString(points)

def get_xsections(geom, xsection_length, shift=1):
    """
    Method to generate perpindicular lines at each point on a provided line. 
    Note that the perpindicular lines generated are placed centrally on the
    provided line, meaning the generated lines will be placed such that they
    are bisected by the provided line.
    Params:
    geom: the geometry/linestring to create perpindicular lines for
    xsection_length: the length of generted perpindicular lines, 
                        in units of the CRS of the geometry
    Returns:
    xsections: list of LineStrings, the cross section lines
    xsection_length: length of each cross-section line that will be created. length
                        in units of the geometry CRS.
    shift: Distance in number of points between two points that form the chord necessary
            to find a perpindicular line. Lower shift means more cross-sections created 
            but the cross-section lines will show greater variety in orientation. Higher 
            shift means fewer cross-sections created but cross-section lines will show
            variations in orientation that are smoothed (effectively by a sliding window).
    """
    assert type(geom) is LineString, "Provided geometry must be of type LineString"
    def get_cross_branch(point1, point2, l):
        """
        """
        def angle_of_connecting_line(point1, point2):
            vec_x = point2[0] - point1[0]
            vec_y = point2[1] - point1[1]
            return np.arctan2(vec_y, vec_x)
        angle_of_segment = angle_of_connecting_line(point1, point2)

        #get angle of direction of the to-be-created segment
        angle_of_xsection = angle_of_segment + np.pi/2

        #generate points on either side point1
        xsegment = LineString([
            Point(
                point1[0] - l*np.cos(angle_of_xsection),
                point1[1] - l*np.sin(angle_of_xsection)
            ),
            Point(
                point1[0] + l*np.cos(angle_of_xsection),
                point1[1] + l*np.sin(angle_of_xsection)
            )
        ])
        return xsegment

    xsections = []
    for p1, p2 in zip(geom.coords[::shift], geom.coords[1::shift], ):
        xsections.append(get_cross_branch(p1, p2, xsection_length))
    xsections.append(get_cross_branch(p2, p1, xsection_length))
    return xsections


def get_valley_lowline(approx_valley_centerline, dem_fn, xsection_length = 300, xsection_complexity=300):
    """
    get valley lowline automatically. generates cross sections for you. this is deprecated, it is better
    to generate your own xsection geometries and use the method below, get_valley_centerline_from_xsections
    """
    cross_sections = get_xsections(approx_valley_centerline, xsection_length)
    complex_cross_sections = [
        increase_line_complexity(x, xsection_complexity) for x in cross_sections
    ]
    xsection_profiles = [
        get_raster_values_for_line(x, dem_fn) for x in complex_cross_sections
    ]
    xsection_profiles_df = pd.DataFrame()
    for index, profile in enumerate(xsection_profiles):
        profile['id'] = index
        xsection_profiles_df = xsection_profiles_df.append(profile)
    xsection_profiles_df = xsection_profiles_df.reset_index(drop=True)
    #drop na in case we queried outside the bounds of the raster
    xsection_profiles_df = xsection_profiles_df.dropna()
    min_elevation_points_df = xsection_profiles_df.loc[xsection_profiles_df.groupby('id')['raster_value'].idxmin()]
    min_elev_line = LineString(list(min_elevation_points_df.apply(lambda row: Point(row.X, row.Y), axis=1)))
    return min_elev_line, cross_sections
    
    
    
    
def get_valley_centerline_from_xsections(xsection_file, dem_file):
    """
    Pretty much just an improved version of the function above (get_valley_lowline_from_xsections).
    This function is more robust to xsection geometries that do not intersect valid data in the provided 
    dem_file.
    The "path distance" or "upstream distance" is calcualted using centroids of the provided cross-
    sections and 
    thus does not depend on the availability of elevation data.
    
    Note: It is assumed that the CRS of the provided geometries and the provided DEM are the same.
    
    Params:
    ========
    xsection_file: str - file path of shapefile or geojson file that contains a series of crosssection
        geometries. NOTE: GEOMETRIES MUST BE IN ORDER FROM DOWNSTREAM TO UPSTREAM.
    dem_file: str - file path of DEM.
    
    Returns:
    ========
    A GeoDataFrame with Point geometries which correspond to the locations of the minimum elevation 
    values, and columns 
    """
    # Read file
    xsection_gdf = gpd.read_file(xsection_file)

    # Set an `id` column with the index which represents the order of creation of the xsection lines (downstream first, upstream last)
    xsection_gdf['id'] = xsection_gdf.index

    # Find centroids of the xsection geometries
    xsection_gdf['xsection_centroid'] = xsection_gdf.geometry.apply(lambda x: x.centroid)

    # Calculate an upstream distance for each xsection using the path formed by the centroids of all cross-sections
    # NOTE that in some older versions of geopandas, the `shift` function below strips away CRS metadata and thus a warning is thrown.
    xsection_gdf['Upstream Distance'] = xsection_gdf['xsection_centroid'].distance(
        gpd.GeoSeries(xsection_gdf['xsection_centroid'].shift(1), crs=xsection_gdf.crs)
    ).fillna(0).cumsum()

    # Increase the complexity of the xsection geometries for superior sampling from the DEM
    xsection_gdf.geometry = xsection_gdf.geometry.apply(
        lambda x: increase_line_complexity(x, int(x.length))
    )

    # Sample from the DEM (fills a column in our dataframe with dataframes)
    xsection_gdf['xsection_raster_values'] = xsection_gdf.geometry.apply(
        lambda x: get_raster_values_for_line(x,dem_file)
    )

    # Find the min raster value for each xsection
    def get_min_value_data(df):
        df = df.dropna()
        if len(df) == 0:
            return np.nan, np.nan, np.nan
        else:
            min_row = df.loc[df['raster_value'].dropna().idxmin()]
            return min_row['X'], min_row['Y'], min_row['raster_value']

    (
        xsection_gdf['X'],
        xsection_gdf['Y'],
        xsection_gdf['Elevation']
    ) = zip(
        *xsection_gdf['xsection_raster_values'].apply(get_min_value_data)
    )
    
    result_gdf = xsection_gdf.drop(['geometry', 'xsection_centroid', 'xsection_raster_values'], axis=1)
    result_gdf.geometry = gpd.points_from_xy(result_gdf.X, result_gdf.Y)
    return result_gdf