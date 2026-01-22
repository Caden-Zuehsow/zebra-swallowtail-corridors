import os
import requests
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.warp
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu
from shapely.geometry import Point
import random

# -----------------------------
# FILE PATHS
# -----------------------------
CORRIDOR_RASTER = "outputs/corridors.tif"
BROWN_COUNTY_SHAPE = "tl_2024_us_county.zip"

# iNaturalist taxon IDs
TAXA = {
    "pawpaw": 50897,
    "zebra_swallowtail": 83086
}
PLACE_ID = 282  # Brown County, IN

TARGET_CRS = "EPSG:26916"  # UTM Zone 16N (meters)

# -----------------------------
# DOWNLOAD US COUNTIES IF MISSING
# -----------------------------
if not os.path.exists(BROWN_COUNTY_SHAPE):
    print("Downloading US counties shapefile...")
    url = "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
    r = requests.get(url, stream=True)
    with open(BROWN_COUNTY_SHAPE, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

# -----------------------------
# LOAD & REPROJECT CORRIDOR RASTER
# -----------------------------
with rasterio.open(CORRIDOR_RASTER) as src:
    transform, width, height = rasterio.warp.calculate_default_transform(
        src.crs, TARGET_CRS, src.width, src.height, *src.bounds
    )

    meta = src.meta.copy()
    meta.update({
        "crs": TARGET_CRS,
        "transform": transform,
        "width": width,
        "height": height
    })

    corridor = np.zeros((height, width), dtype=np.float32)

    rasterio.warp.reproject(
        source=rasterio.band(src, 1),
        destination=corridor,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=TARGET_CRS,
        resampling=rasterio.enums.Resampling.nearest
    )

pixel_size = transform.a  # meters
crs = TARGET_CRS

# -----------------------------
# DISTANCE TO CORRIDOR (METERS)
# -----------------------------
corridor = np.where(np.isfinite(corridor), corridor, 0)

if np.count_nonzero(corridor) == 0:
    raise ValueError("Corridor raster contains no nonzero values.")

threshold = np.percentile(corridor[corridor > 0], 90)
corridor_bin = (corridor >= threshold).astype(np.uint8)

distance_to_corridor = distance_transform_edt(1 - corridor_bin) * pixel_size

# -----------------------------
# LOAD BROWN COUNTY
# -----------------------------
counties = gpd.read_file(BROWN_COUNTY_SHAPE)
brown_county = counties[counties["GEOID"] == "18013"].to_crs(crs)

if brown_county.empty:
    raise ValueError("Brown County not found.")

county_polygon = brown_county.geometry.iloc[0]

# -----------------------------
# FETCH iNATURALIST OBSERVATIONS
# -----------------------------
def fetch_inat_obs(taxon_id, place_id=PLACE_ID):
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "place_id": place_id,
        "quality_grade": "research",
        "verifiable": "true",
        "spam": "false",
        "per_page": 200,
        "page": 1
    }

    results_all = []
    while True:
        r = requests.get(url, params=params)
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        results_all.extend(results)
        params["page"] += 1

    records = []
    for obs in results_all:
        geo = obs.get("geojson")
        if geo:
            lon, lat = geo["coordinates"]
            records.append((lon, lat))

    if not records:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(
            [x for x, y in records],
            [y for x, y in records]
        ),
        crs="EPSG:4326"
    ).to_crs(crs)

    return gdf[gdf.geometry.within(county_polygon)]

# -----------------------------
# FETCH DATA
# -----------------------------
pawpaw_points = fetch_inat_obs(TAXA["pawpaw"])
butterfly_points = fetch_inat_obs(TAXA["zebra_swallowtail"])

print(f"Pawpaw points: {len(pawpaw_points)}")
print(f"Zebra swallowtail points: {len(butterfly_points)}")

# -----------------------------
# SAMPLE DISTANCES
# -----------------------------
def sample_distances(gdf):
    distances = []
    for pt in gdf.geometry:
        col, row = ~transform * (pt.x, pt.y)
        row, col = int(row), int(col)
        if (
            0 <= row < distance_to_corridor.shape[0]
            and 0 <= col < distance_to_corridor.shape[1]
        ):
            distances.append(distance_to_corridor[row, col])
    return np.array(distances)

obs_distances = sample_distances(butterfly_points)

# -----------------------------
# NULL MODEL (RANDOM POINTS)
# -----------------------------
minx, miny, maxx, maxy = county_polygon.bounds
random_points = []

while len(random_points) < len(obs_distances):
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    p = Point(x, y)
    if county_polygon.contains(p):
        random_points.append(p)

random_gdf = gpd.GeoDataFrame(geometry=random_points, crs=crs)
rand_distances = sample_distances(random_gdf)

# -----------------------------
# STATISTICAL TEST
# -----------------------------
MIN_POINTS = 3

if len(obs_distances) < MIN_POINTS or len(rand_distances) < MIN_POINTS:
    stat, p = np.nan, np.nan
else:
    stat, p = mannwhitneyu(
        obs_distances,
        rand_distances,
        alternative="less",
        method="auto"
    )
    assert 0 <= p <= 1, f"Invalid p-value: {p}"

# -----------------------------
# RESULTS
# -----------------------------
print("\nVALIDATION RESULTS")
print("------------------")
print(f"Observed median distance: {np.median(obs_distances):.2f} m")
print(f"Random median distance:   {np.median(rand_distances):.2f} m")
print(f"P-value:                 {p:.3e}")
