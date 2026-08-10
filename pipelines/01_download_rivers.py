"""
src/01_download_rivers.py - VERSION 2 (Full Download)
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import ee
import json

OUTPUT_DIR = project_root / "data" / "raw" / "osm_kerala"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RIVERS_GEOJSON = OUTPUT_DIR / "kerala_rivers.geojson"
COASTLINE_GEOJSON = OUTPUT_DIR / "kerala_coastline.geojson"

KERALA_DISTRICTS_BBOX = [
    ("Kasargod",         [74.85, 12.20, 75.40, 12.80]),
    ("Kannur",           [75.10, 11.65, 75.90, 12.30]),
    ("Wayanad",          [75.80, 11.45, 76.40, 11.95]),
    ("Kozhikode",        [75.45, 11.10, 76.30, 11.70]),
    ("Malappuram",       [75.85, 10.65, 76.85, 11.30]),
    ("Palakkad",         [76.20, 10.30, 77.10, 11.10]),
    ("Thrissur",         [76.00, 10.10, 76.80, 10.75]),
    ("Ernakulam",        [76.05, 9.65,  76.90, 10.25]),
    ("Idukki",           [76.65, 9.40,  77.30, 10.40]),
    ("Kottayam",         [76.30, 9.15,  76.95, 9.85]),
    ("Alappuzha",        [76.20, 9.05,  76.65, 9.75]),
    ("Pathanamthitta",   [76.50, 9.00,  77.10, 9.55]),
    ("Kollam",           [76.40, 8.65,  77.20, 9.25]),
    ("Thiruvananthapuram", [76.70, 8.20, 77.40, 8.85]),
]


def initialize_gee():
    print("\nInitializing GEE...")
    try:
        ee.Initialize()
        print("GEE initialized")
        return True
    except Exception as e:
        print(f"GEE failed: {e}")
        return False


def download_rivers_per_district():
    print("\nDownloading rivers (district by district)...")
    rivers_dataset = ee.FeatureCollection('WWF/HydroSHEDS/v1/FreeFlowingRivers')
    all_features = []

    for district_name, bbox in KERALA_DISTRICTS_BBOX:
        print(f"   {district_name}...", end=" ")
        try:
            district_geom = ee.Geometry.Rectangle(bbox)
            district_rivers = rivers_dataset.filterBounds(district_geom)
            district_rivers = district_rivers.filter(ee.Filter.lte("RIV_ORD", 7))
            count = district_rivers.size().getInfo()
            print(f"{count} rivers", end=" ")

            if count == 0:
                print("(skipping)")
                continue

            if count > 4500:
                print("\n      Splitting in half...")
                mid_lat = (bbox[1] + bbox[3]) / 2
                top_half = ee.Geometry.Rectangle([bbox[0], mid_lat, bbox[2], bbox[3]])
                bottom_half = ee.Geometry.Rectangle([bbox[0], bbox[1], bbox[2], mid_lat])
                for half_name, half_geom in [("top", top_half), ("bottom", bottom_half)]:
                    half_rivers = rivers_dataset.filterBounds(half_geom).filter(ee.Filter.lte("RIV_ORD", 7))
                    half_data = half_rivers.getInfo()
                    half_features = half_data.get("features", [])
                    all_features.extend(half_features)
                    print(f"      {half_name}: {len(half_features)} segments")
            else:
                data = district_rivers.getInfo()
                features = data.get("features", [])
                all_features.extend(features)
                print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print("\n   Removing duplicates...")
    seen_ids = set()
    unique_features = []
    for feat in all_features:
        props = feat.get("properties", {})
        riv_id = props.get("HYRIV_ID")
        if riv_id is None:
            unique_features.append(feat)
        elif riv_id not in seen_ids:
            seen_ids.add(riv_id)
            unique_features.append(feat)

    print(f"   {len(all_features)} total -> {len(unique_features)} unique")
    final_geojson = {"type": "FeatureCollection", "features": unique_features}
    with open(RIVERS_GEOJSON, "w", encoding="utf-8") as f:
        json.dump(final_geojson, f)
    size_mb = RIVERS_GEOJSON.stat().st_size / (1024 * 1024)
    print(f"   Saved ({size_mb:.1f} MB)")
    return len(unique_features)


def download_coastline():
    print("\nCreating Kerala coastline...")
    coastline_coords = [
        [74.86, 12.79], [75.15, 12.50], [75.30, 12.00], [75.50, 11.80],
        [75.70, 11.50], [75.85, 11.20], [75.95, 10.90], [76.05, 10.50],
        [76.15, 10.20], [76.20, 9.95],  [76.25, 9.70],  [76.30, 9.45],
        [76.35, 9.20],  [76.40, 8.90],  [76.55, 8.60],  [76.85, 8.30],
    ]
    try:
        coastline_geom = ee.Geometry.LineString(coastline_coords)
        coastline_fc = ee.FeatureCollection([ee.Feature(coastline_geom, {"type": "coast"})])
        geojson_dict = coastline_fc.getInfo()
        with open(COASTLINE_GEOJSON, "w", encoding="utf-8") as f:
            json.dump(geojson_dict, f, indent=2)
        print("   Saved coastline")
        return True
    except Exception as e:
        print(f"   Failed: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("STEP 1 V2: DOWNLOAD ALL KERALA RIVERS")
    print("="*60)
    if not initialize_gee():
        return
    river_count = download_rivers_per_district()
    download_coastline()
    print("\n" + "="*60)
    if river_count > 1000:
        print(f"DONE: {river_count} rivers downloaded")
    else:
        print(f"WARNING: Only {river_count} rivers (expected ~9000)")
    print("="*60)


if __name__ == "__main__":
    main()
