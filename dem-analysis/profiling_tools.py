import numpy as np
import shapely
import geopandas as gpd
import rasterio
import pandas as pd
    

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
    df['path_distance'] = pd.Series(gdf.distance(gdf.shift(1)).fillna(0)).cumsum()
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