import geopandas as gpd
import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu
from shapely.geometry import Point
import random

# -----------------------------
# FILE PATHS
# -----------------------------
CORRIDOR_RASTER = "outputs/corridors.tif"
BUTTERFLY_FILE = "data/zebra_swallowtail_points.shp"

# -----------------------------
# LOAD CORRIDOR RASTER
# -----------------------------
with rasterio.open(CORRIDOR_RASTER) as src:
    corridor = src.read(1)
    transform = src.transform
    crs = src.crs
    pixel_size = src.res[0]

# Remove nodata
corridor = np.where(np.isfinite(corridor), corridor, 0)

# Threshold top 10%
threshold = np.percentile(corridor[corridor > 0], 90)
corridor_bin = (corridor >= threshold).astype(np.uint8)

# Distance raster
distance_to_corridor = distance_transform_edt(1 - corridor_bin) * pixel_size

# -----------------------------
# LOAD BUTTERFLIES (SYNTHETIC)
# -----------------------------
butterflies = gpd.read_file(BUTTERFLY_FILE).to_crs(crs)

# Check for empty geometries
empty_count = butterflies.geometry.is_empty.sum()
if empty_count > 0:
    print(f"Warning: {empty_count} empty geometries found in butterfly shapefile, skipping them.")
butterflies = butterflies[~butterflies.geometry.is_empty]

if len(butterflies) == 0:
    raise ValueError("No valid butterfly points to validate. Check your shapefile!")

# -----------------------------
# SAMPLE DISTANCE FUNCTION
# -----------------------------
def sample_distances(gdf):
    distances = []
    for pt in gdf.geometry:
        if pt is None or pt.is_empty:
            continue  # skip invalid points
        col, row = ~transform * (pt.x, pt.y)
        row, col = int(row), int(col)
        if 0 <= row < distance_to_corridor.shape[0] and 0 <= col < distance_to_corridor.shape[1]:
            distances.append(distance_to_corridor[row, col])
    return np.array(distances)

# -----------------------------
# OBSERVED DISTANCES
# -----------------------------
obs_distances = sample_distances(butterflies)

# -----------------------------
# RANDOM POINTS (NULL MODEL)
# -----------------------------
minx, miny, maxx, maxy = butterflies.total_bounds
random_points = []

while len(random_points) < len(obs_distances):
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    random_points.append(Point(x, y))

random_gdf = gpd.GeoDataFrame(geometry=random_points, crs=crs)
rand_distances = sample_distances(random_gdf)

# -----------------------------
# STAT TEST
# -----------------------------
stat, p = mannwhitneyu(obs_distances, rand_distances, alternative="less")

print("VALIDATION RESULTS")
print("------------------")
print(f"Observed median distance: {np.median(obs_distances):.1f} m")
print(f"Random median distance:   {np.median(rand_distances):.1f} m")
print(f"P-value:                 {p:.4f}")
