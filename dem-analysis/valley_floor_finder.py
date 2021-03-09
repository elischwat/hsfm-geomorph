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

import shapely
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Point
import contextily as ctx

profiles_shapefile = "/data2/elilouis/hsfm-geomorph/data/profiles/profiles.shp"
gdf = gpd.read_file(profiles_shapefile).to_crs(epsg='32610')

geom = gdf.geometry.iloc[5]
geom


def get_xsections(geom, xsection_length):
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
    for p1, p2 in zip(geom.coords, geom.coords[1:]):
        xsections.append(get_cross_branch(p1, p2, xsection_length))
    xsections.append(get_cross_branch(p2, p1, xsection_length))
    return xsections


def plot_geoms(geoms_list):
    gdf = gpd.GeoDataFrame(geometry=geoms_list)
    gdf = gdf.set_crs(epsg=32610)
    gdf = gdf.to_crs(epsg=3857)
    ax = gdf.plot()
    ctx.add_basemap(ax, source=ctx.providers.OpenTopoMap)
    return ax


ax = plot_geoms(get_xsections(geom, 300) + [geom])
plt.show()

ax2 = plot_geoms([geom])
ax2.set_xlim(ax.get_xlim())
ax2.set_ylim(ax.get_ylim())
plt.show()


