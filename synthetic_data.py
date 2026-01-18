"""
Synthetic Data Generator for Zebra Swallowtail Corridor Model
------------------------------------------------------------

Creates fake-but-structured data for testing:
- Pawpaw groves (points)
- Rivers (lines)
- Landcover raster
- Zebra swallowtail observations (biased toward corridors)

All data are projected (meters) and GIS-ready.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, LineString
import os

np.random.seed(42)

os.makedirs("data", exist_ok=True)

CRS = "EPSG:3857"  # meters
EXTENT = 10000     # 10 km x 10 km

# --------------------------------------------------
# 1. Pawpaw groves (clustered)
# --------------------------------------------------
pawpaw_points = []

cluster_centers = [
    (2000, 8000),
    (5000, 5000),
    (8000, 2000)
]

for cx, cy in cluster_centers:
    for _ in range(30):
        x = np.random.normal(cx, 300)
        y = np.random.normal(cy, 300)
        pawpaw_points.append(Point(x, y))

pawpaw = gpd.GeoDataFrame(
    geometry=pawpaw_points,
    crs=CRS
)

pawpaw.to_file("data/pawpaw_points.shp")

# --------------------------------------------------
# 2. Rivers (linear navigation features)
# --------------------------------------------------
river_lines = [
    LineString([(0, 9000), (10000, 1000)]),
    LineString([(0, 3000), (10000, 7000)])
]

rivers = gpd.GeoDataFrame(
    geometry=river_lines,
    crs=CRS
)

rivers.to_file("data/rivers.shp")

# --------------------------------------------------
# 3. Landcover raster
# --------------------------------------------------
width = height = 100
pixel_size = EXTENT / width

transform = from_origin(0, EXTENT, pixel_size, pixel_size)

landcover = np.ones((height, width), dtype="uint8") * 81  # Agriculture

# Forest along rivers
for i in range(height):
    for j in range(width):
        x = j * pixel_size
        y = EXTENT - i * pixel_size
        if abs(y - (9000 - 0.8 * x)) < 400:
            landcover[i, j] = 41  # Forest

meta = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": CRS,
    "transform": transform
}

with rasterio.open("data/landcover.tif", "w", **meta) as dst:
    dst.write(landcover, 1)

# --------------------------------------------------
# 4. Zebra swallowtail observations (biased)
# --------------------------------------------------
swallowtail_points = []

for _ in range(250):
    river = river_lines[np.random.randint(0, len(river_lines))]
    position = river.interpolate(np.random.random(), normalized=True)
    offset = np.random.normal(0, 200)
    swallowtail_points.append(position.buffer(offset).centroid)

swallowtails = gpd.GeoDataFrame(
    geometry=swallowtail_points,
    crs=CRS
)

swallowtails.to_file("data/zebra_swallowtail_points.shp")

print("Synthetic data generation complete.")
