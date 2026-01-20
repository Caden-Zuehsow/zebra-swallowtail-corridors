import os
import requests
import pandas as pd
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
BROWN_COUNTY_SHAPE = "tl_2024_us_county.zip"  # Will auto-download if missing

# iNaturalist taxon IDs
TAXA = {
    "pawpaw": 50897,            # Asimina triloba
    "zebra_swallowtail": 83086  # Eurytides marcellus
}
PLACE_ID = 282  # Brown County, IN

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
# LOAD CORRIDOR RASTER
# -----------------------------
with rasterio.open(CORRIDOR_RASTER) as src:
    corridor = src.read(1)
    transform = src.transform
    crs = src.crs
    pixel_size = src.res[0]

# Remove nodata and threshold top 10%
corridor = np.where(np.isfinite(corridor), corridor, 0)
threshold = np.percentile(corridor[corridor > 0], 90)
corridor_bin = (corridor >= threshold).astype(np.uint8)
distance_to_corridor = distance_transform_edt(1 - corridor_bin) * pixel_size

# -----------------------------
# LOAD BROWN COUNTY
# -----------------------------
gdf = gpd.read_file(BROWN_COUNTY_SHAPE)
brown_county = gdf[gdf['GEOID'] == '18013'].to_crs(crs)
if brown_county.empty:
    raise ValueError("Brown County (GEOID 18013) not found in shapefile.")
county_polygon = brown_county.geometry.iloc[0]

# -----------------------------
# HELPER: FETCH OBSERVATIONS FROM iNaturalist
# -----------------------------
def fetch_inat_obs(taxon_id, place_id=PLACE_ID):
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "place_id": place_id,
        "quality_grade": "research",  # Only research-grade
        "verifiable": "true",
        "spam": "false",
        "per_page": 200,
        "page": 1
    }
    all_results = []
    while True:
        r = requests.get(url, params=params)
        data = r.json()
        results = data.get('results', [])
        if not results:
            break
        all_results.extend(results)
        params['page'] += 1

    records = []
    for obs in all_results:
        geo = obs.get('geojson')
        if geo:
            lon, lat = geo['coordinates']
            records.append({
                "id": obs['id'],
                "observed_on": obs['observed_on'],
                "latitude": lat,
                "longitude": lon,
                "quality_grade": obs['quality_grade'],
                "species_guess": obs['species_guess'],
                "scientific_name": obs['taxon']['name'] if obs.get('taxon') else None
            })
    df = pd.DataFrame(records)
    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"
    ).to_crs(crs)
    # Keep only points inside county
    gdf = gdf[gdf.geometry.within(county_polygon)]
    return gdf

# -----------------------------
# FETCH BOTH TAXA
# -----------------------------
pawpaw_points = fetch_inat_obs(TAXA['pawpaw'])
butterfly_points = fetch_inat_obs(TAXA['zebra_swallowtail'])

print(f"Pawpaw points in county: {len(pawpaw_points)}")
print(f"Zebra swallowtail points in county: {len(butterfly_points)}")

# -----------------------------
# SAMPLE DISTANCE FUNCTION
# -----------------------------
def sample_distances(gdf):
    distances = []
    for pt in gdf.geometry:
        if pt is None or pt.is_empty:
            continue
        col, row = ~transform * (pt.x, pt.y)
        row, col = int(row), int(col)
        if 0 <= row < distance_to_corridor.shape[0] and 0 <= col < distance_to_corridor.shape[1]:
            distances.append(distance_to_corridor[row, col])
    return np.array(distances)

# -----------------------------
# OBSERVED DISTANCES
# -----------------------------
obs_distances = sample_distances(butterfly_points)

# -----------------------------
# RANDOM POINTS (NULL MODEL)
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
# STAT TEST
# -----------------------------
MIN_POINTS = 3
if len(obs_distances) < MIN_POINTS or len(rand_distances) < MIN_POINTS:
    print("Warning: Not enough points to perform Mann–Whitney test.")
    stat, p = np.nan, np.nan
else:
    stat, p = mannwhitneyu(obs_distances, rand_distances, alternative="less")

# -----------------------------
# RESULTS
# -----------------------------
print("VALIDATION RESULTS")
print("------------------")
print(f"Observed median distance: {np.median(obs_distances) if len(obs_distances) > 0 else 'nan'} m")
print(f"Random median distance:   {np.median(rand_distances) if len(rand_distances) > 0 else 'nan'} m")
print(f"P-value:                 {p if not np.isnan(p) else 'nan'}")
