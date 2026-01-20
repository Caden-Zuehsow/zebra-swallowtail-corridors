"""
Zebra Swallowtail Corridor Modeling
----------------------------------
Models movement corridors using pawpaw host plants
and riparian navigation zones.
"""

import os
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize, shapes
import rasterio
from rasterio.transform import from_origin
from skimage.graph import route_through_array
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape, Point

# -------------------------
# CREATE OUTPUT DIRECTORY
# -------------------------
os.makedirs("outputs", exist_ok=True)

# -------------------------
# FILE PATHS
# -------------------------
PAWPAW = "data/pawpaw_points.shp"
RIVERS = "data/rivers_lines.shp"  # OSM rivers/streams
LAKES = "data/lakes_polys.shp"    # OSM lakes/reservoirs
COUNTIES = "data/tl_2024_us_county.zip"

OUT_RESISTANCE = "outputs/resistance.tif"
OUT_CORRIDORS = "outputs/corridors.tif"
OUT_CORRIDOR_SHP = "outputs/corridors.shp"

# -------------------------
# LOAD DATA
# -------------------------
pawpaw = gpd.read_file(PAWPAW)
rivers = gpd.read_file(RIVERS)
lakes  = gpd.read_file(LAKES)
counties = gpd.read_file(COUNTIES)

brown = counties[counties["GEOID"] == "18013"].to_crs(pawpaw.crs)
county_geom = brown.geometry.iloc[0]

# -------------------------
# SIMPLIFY AND CLIP FEATURES
# -------------------------
rivers["geometry"] = rivers.geometry.simplify(tolerance=10)
lakes["geometry"]  = lakes.geometry.simplify(tolerance=10)

rivers = rivers[rivers.geometry.intersects(county_geom)]
lakes  = lakes[lakes.geometry.intersects(county_geom)]
pawpaw = pawpaw[pawpaw.geometry.within(county_geom)]

# -------------------------
# DEFINE RASTER GRID
# -------------------------
# Pixel size in degrees (~30 m)
pixel_size = 0.0003
minx, miny, maxx, maxy = county_geom.bounds
width = int(np.ceil((maxx - minx) / pixel_size))
height = int(np.ceil((maxy - miny) / pixel_size))
transform = from_origin(minx, maxy, pixel_size, pixel_size)

meta = {
    "driver": "GTiff",
    "dtype": "float32",
    "nodata": None,
    "width": width,
    "height": height,
    "count": 1,
    "crs": pawpaw.crs,
    "transform": transform
}

# -------------------------
# RASTERIZE FEATURES
# -------------------------
def rasterize_gdf(gdf):
    return rasterize(
        [(geom, 1) for geom in gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype="uint8"
    )

pawpaw_raster = rasterize_gdf(pawpaw)
river_raster  = rasterize_gdf(rivers)
lake_raster   = rasterize_gdf(lakes)

# -------------------------
# DISTANCE SURFACES (meters)
# -------------------------
pawpaw_dist = distance_transform_edt(1 - pawpaw_raster) * 30
river_dist  = distance_transform_edt(1 - river_raster) * 30
lake_dist   = distance_transform_edt(1 - lake_raster) * 30

# -------------------------
# LAND COST (uniform)
# -------------------------
land_cost = np.ones_like(pawpaw_raster, dtype="float32") * 50

# -------------------------
# COMBINE INTO RESISTANCE
# -------------------------
resistance = (
    0.6 * pawpaw_dist +    # pawpaw distance
    0.8 * river_dist +     # rivers/streams distance
    0.5 * lake_dist +      # lakes/reservoirs distance
    1.0 * land_cost        # land cost
)

resistance = np.clip(resistance, 1, None)

with rasterio.open(OUT_RESISTANCE, "w", **meta) as dst:
    dst.write(resistance, 1)

# -------------------------
# LEAST-COST CORRIDORS
# -------------------------
def to_rc(pt):
    col, row = ~transform * (pt.x, pt.y)
    return int(row), int(col)

corridor = np.zeros_like(resistance)
nodes = pawpaw.geometry.sample(min(20, len(pawpaw)))

for i in range(len(nodes) - 1):
    start = to_rc(nodes.iloc[i])
    end = to_rc(nodes.iloc[i + 1])
    path, _ = route_through_array(resistance, start, end, fully_connected=True)
    for r, c in path:
        corridor[r, c] += 1

with rasterio.open(OUT_CORRIDORS, "w", **meta) as dst:
    dst.write(corridor.astype("float32"), 1)

# -------------------------
# EXPORT CORRIDOR SHAPEFILE
# -------------------------
features = []

for geom, value in shapes(corridor, mask=corridor > 0, transform=transform):
    features.append({"geometry": shape(geom), "value": value})

gpd.GeoDataFrame(features, crs=pawpaw.crs).to_file(OUT_CORRIDOR_SHP)

print("Corridor modeling complete.")

