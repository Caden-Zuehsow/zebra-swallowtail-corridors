import os
import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import osmnx as ox

# -----------------------------
# FILE PATHS
# -----------------------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

BROWN_COUNTY_SHAPE = os.path.join(DATA_DIR, "tl_2024_us_county.zip")
RIVERS_FILE = os.path.join(DATA_DIR, "rivers_lines.shp")  # LINESTRING rivers
LAKES_FILE = os.path.join(DATA_DIR, "lakes_polys.shp")    # POLYGON lakes/reservoirs
PAWPAW_FILE = os.path.join(DATA_DIR, "pawpaw_points.shp")
BUTTERFLY_FILE = os.path.join(DATA_DIR, "zebra_swallowtail_points.shp")

BROWN_GEOID = "18013"
PLACE_ID = 282  # Brown County, IN

# -----------------------------
# DOWNLOAD US COUNTIES
# -----------------------------
if not os.path.exists(BROWN_COUNTY_SHAPE):
    print("Downloading US counties shapefile...")
    url_counties = "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
    r = requests.get(url_counties)
    with open(BROWN_COUNTY_SHAPE, "wb") as f:
        f.write(r.content)
    print("Counties downloaded.")

counties = gpd.read_file(BROWN_COUNTY_SHAPE)
brown = counties[counties['GEOID'] == BROWN_GEOID]
if brown.empty:
    raise ValueError("Brown County not found in shapefile!")
brown = brown.to_crs("EPSG:4326")
brown_poly = brown.geometry.iloc[0]

# -----------------------------
# DOWNLOAD OSM WATERWAYS
# -----------------------------
if not os.path.exists(RIVERS_FILE):
    print("Fetching OSM waterways for Brown County...")
    tags = {"waterway": True, "natural": ["water", "wetland"]}

    # Download all OSM features within the county polygon
    rivers_gdf = ox.features_from_polygon(brown_poly, tags=tags)
    rivers_gdf = rivers_gdf[rivers_gdf.geometry.notna()]  # remove empty geometries
    rivers_gdf = rivers_gdf.to_crs("EPSG:4326")

    # Separate LINESTRING rivers/streams from POLYGON lakes/reservoirs
    rivers_lines = rivers_gdf[rivers_gdf.geometry.type.isin(["LineString", "MultiLineString"])]
    rivers_polys = rivers_gdf[rivers_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    # Clip precisely to Brown County
    rivers_lines = rivers_lines[rivers_lines.geometry.intersects(brown_poly)]
    rivers_polys = rivers_polys[rivers_polys.geometry.intersects(brown_poly)]

    # Save shapefiles
    rivers_lines.to_file(RIVERS_FILE)
    rivers_polys.to_file(LAKES_FILE)
    print(f"Saved {len(rivers_lines)} rivers/streams to {RIVERS_FILE}")
    print(f"Saved {len(rivers_polys)} lakes/reservoirs to {LAKES_FILE}")
else:
    print("Rivers file already exists, skipping download.")

# -----------------------------
# FETCH iNaturalist DATA
# -----------------------------
def fetch_inat(taxon_id, output_file, county_polygon):
    print(f"Fetching iNaturalist data for taxon {taxon_id}...")
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "taxon_id": taxon_id,
        "place_id": PLACE_ID,
        "quality_grade": "research",
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
            records.append({"longitude": lon, "latitude": lat, "id": obs['id']})

    if not records:
        print(f"No observations found for taxon {taxon_id}")
        return

    df = pd.DataFrame(records)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
    )

    # Clip to Brown County polygon
    gdf = gdf[gdf.geometry.within(county_polygon)]

    gdf.to_file(output_file)
    print(f"Saved {len(gdf)} observations to {output_file}")

# -----------------------------
# FETCH PAWPAW AND BUTTERFLY
# -----------------------------
fetch_inat(50897, PAWPAW_FILE, brown_poly)       # Pawpaw
fetch_inat(83086, BUTTERFLY_FILE, brown_poly)   # Zebra Swallowtail
