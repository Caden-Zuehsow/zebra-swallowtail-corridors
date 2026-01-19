import geopandas as gpd
import rasterio
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu
from shapely.geometry import Point
import random

# -----------------------------
# FILE PATHS (EDIT THESE)
# -----------------------------
CORRIDOR_RASTER = "outputs/corridors.tif"
BUTTERFLY_CSV = "data/zebra_swallowtail_observations.csv"
BOUNDARY_FILE = "data/brown_county_boundary.gpkg"

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
# LOAD BUTTERFLIES
# -----------------------------
butterflies = gpd.read_file(BUTTERFLY_CSV)
butterflies = butterflies.set_geometry(
    gpd.points_from_xy(butterflies.longitude, butterflies.latitude),
    crs="EPSG:4326"
).to_crs(crs)

# -----------------------------
# LOAD BOUNDARY
# -----------------------------
boundary = gpd.read_file(BOUNDARY_FILE).to_crs(crs)

# -----------------------------
# SAMPLE DISTANCE FUNCTION
# -----------------------------
def sample_distances(gdf):
    distances = []
    for pt in gdf.geometry:
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
minx, miny, maxx, maxy = boundary.total_bounds
random_points = []

while len(random_points) < len(obs_distances):
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    p = Point(x, y)
    if boundary.contains(p).any():
        random_points.append(p)

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
