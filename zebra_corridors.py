"""
Zebra Swallowtail Corridor Modeling
----------------------------------
Models movement corridors using pawpaw host plants
and riparian navigation zones.

Outputs:
- resistance.tif
- corridors.tif
- corridors.shp
"""

import os  # <-- ADD THIS
import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import rasterize, shapes
from scipy.ndimage import distance_transform_edt
from skimage.graph import route_through_array
from shapely.geometry import shape

# -------------------------
# CREATE OUTPUT DIRECTORY
# -------------------------
os.makedirs("outputs", exist_ok=True)  # <-- ADD THIS

# -------------------------
# FILE PATHS
# -------------------------
PAWPAW = "data/pawpaw_points.shp"
RIVERS = "data/rivers.shp"
LANDCOVER = "data/landcover.tif"

OUT_RESISTANCE = "outputs/resistance.tif"
OUT_CORRIDORS = "outputs/corridors.tif"
OUT_CORRIDOR_SHP = "outputs/corridors.shp"

# -------------------------
# LOAD DATA
# -------------------------
pawpaw = gpd.read_file(PAWPAW)
rivers = gpd.read_file(RIVERS)

with rasterio.open(LANDCOVER) as src:
    landcover = src.read(1)
    transform = src.transform
    meta = src.meta.copy()

# -------------------------
# RASTERIZE FEATURES
# -------------------------
pawpaw_raster = rasterize(
    [(g, 1) for g in pawpaw.geometry],
    out_shape=landcover.shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

river_raster = rasterize(
    [(g, 1) for g in rivers.geometry],
    out_shape=landcover.shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# -------------------------
# DISTANCE SURFACES
# -------------------------
pawpaw_dist = distance_transform_edt(1 - pawpaw_raster)
river_dist = distance_transform_edt(1 - river_raster)

# -------------------------
# LAND COVER RESISTANCE
# -------------------------
landcover_costs = {
    11: 5,    # Water
    21: 50,   # Developed
    41: 5,    # Forest
    52: 15,   # Shrub
    71: 20,   # Grass
    81: 25,   # Agriculture
    90: 5     # Wetlands
}

land_cost = np.ones_like(landcover, dtype="float32") * 50
for code, cost in landcover_costs.items():
    land_cost[landcover == code] = cost

# -------------------------
# COMBINE INTO RESISTANCE
# -------------------------
w_pawpaw = 0.6
w_river = 0.8
w_land = 1.0

resistance = (
    w_pawpaw * pawpaw_dist +
    w_river * river_dist +
    w_land * land_cost
)

resistance = np.clip(resistance, 1, None)

meta.update(dtype="float32", count=1)

with rasterio.open(OUT_RESISTANCE, "w", **meta) as dst:
    dst.write(resistance, 1)

# -------------------------
# LEAST-COST CORRIDORS
# -------------------------
def to_rc(point):
    row, col = ~transform * (point.x, point.y)
    return int(row), int(col)

corridor = np.zeros_like(resistance)

nodes = pawpaw.geometry.sample(min(20, len(pawpaw)))

for i in range(len(nodes) - 1):
    start = to_rc(nodes.iloc[i])
    end = to_rc(nodes.iloc[i + 1])

    path, _ = route_through_array(
        resistance,
        start,
        end,
        fully_connected=True
    )

    for r, c in path:
        corridor[r, c] += 1

with rasterio.open(OUT_CORRIDORS, "w", **meta) as dst:
    dst.write(corridor.astype("float32"), 1)

# -------------------------
# EXPORT TO SHAPEFILE
# -------------------------
features = []

for geom, value in shapes(corridor, mask=corridor > 0, transform=transform):
    features.append({
        "geometry": shape(geom),
        "value": value
    })

gpd.GeoDataFrame(features, crs=pawpaw.crs).to_file(OUT_CORRIDOR_SHP)

print("Corridor modeling complete.")

