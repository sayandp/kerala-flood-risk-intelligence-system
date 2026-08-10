"""
============================================================
src/07_download_osm_waterways.py
============================================================
Downloads ALL OSM waterways for Kerala (rivers, canals, streams,
ditches, drains, kayals, backwaters - everything water-related).

Uses Overpass API (free, no signup required).

OUTPUT: data/raw/osm_kerala/kerala_waterways_osm.geojson
        Then re-runs distance calculation with this complete dataset.
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import requests
import json
import time

OUTPUT_DIR = project_root / "data" / "raw" / "osm_kerala"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "kerala_waterways_osm.geojson"


# Kerala districts (smaller bounding boxes = faster downloads)
KERALA_DISTRICTS = [
    ("Kasargod",            74.85, 12.20, 75.40, 12.80),
    ("Kannur",              75.10, 11.65, 75.90, 12.30),
    ("Wayanad",             75.80, 11.45, 76.40, 11.95),
    ("Kozhikode",           75.45, 11.10, 76.30, 11.70),
    ("Malappuram",          75.85, 10.65, 76.85, 11.30),
    ("Palakkad",            76.20, 10.30, 77.10, 11.10),
    ("Thrissur",            76.00, 10.10, 76.80, 10.75),
    ("Ernakulam",           76.05,  9.65, 76.90, 10.25),
    ("Idukki",              76.65,  9.40, 77.30, 10.40),
    ("Kottayam",            76.30,  9.15, 76.95,  9.85),
    ("Alappuzha",           76.20,  9.05, 76.65,  9.75),
    ("Pathanamthitta",      76.50,  9.00, 77.10,  9.55),
    ("Kollam",              76.40,  8.65, 77.20,  9.25),
    ("Thiruvananthapuram",  76.70,  8.20, 77.40,  8.85),
]


# Overpass query: get ALL water-related features
# This is a "kitchen sink" query that gets everything water
def build_overpass_query(south, west, north, east):
    """Build Overpass QL query for waterways in a bounding box."""
    return f"""
[out:json][timeout:90];
(
  way["waterway"~"river|stream|canal|drain|ditch|brook|stream_intermittent"]({south},{west},{north},{east});
  way["natural"="water"]({south},{west},{north},{east});
  way["water"]({south},{west},{north},{east});
  relation["natural"="water"]({south},{west},{north},{east});
);
out geom;
"""


# Multiple Overpass servers (failover)
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


def query_overpass(query, max_retries=3):
    """Query Overpass API with failover and retries."""
    for attempt in range(max_retries):
        for server in OVERPASS_SERVERS:
            try:
                response = requests.post(
                    server,
                    data={'data': query},
                    timeout=120
                )
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited
                    print(f"      Rate limited on {server[:30]}, waiting 30s...")
                    time.sleep(30)
                    continue
                elif response.status_code == 504:
                    print(f"      Timeout on {server[:30]}, trying next...")
                    continue
                else:
                    print(f"      HTTP {response.status_code} on {server[:30]}")
            except requests.exceptions.Timeout:
                print(f"      Connection timeout on {server[:30]}")
            except Exception as e:
                print(f"      Error: {str(e)[:50]}")
        
        # Wait between retry rounds
        if attempt < max_retries - 1:
            print(f"      Retrying in 10s (attempt {attempt + 2}/{max_retries})...")
            time.sleep(10)
    
    return None


def osm_to_geojson(osm_data):
    """Convert Overpass JSON response to GeoJSON features."""
    features = []
    
    for element in osm_data.get('elements', []):
        if element.get('type') != 'way':
            continue
        
        geometry = element.get('geometry', [])
        if len(geometry) < 2:
            continue
        
        # Build LineString geometry
        coords = [[node['lon'], node['lat']] for node in geometry]
        
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coords
            },
            "properties": {
                "osm_id": element.get('id'),
                **element.get('tags', {})
            }
        }
        features.append(feature)
    
    return features


def main():
    print("\n" + "="*70)
    print("STEP 7: DOWNLOAD OSM WATERWAYS (rivers, canals, streams, lakes)")
    print("="*70)
    print(f"\nDownloading from {len(OVERPASS_SERVERS)} Overpass mirrors with auto-failover")
    print("This takes 5-15 minutes — be patient, the data is comprehensive\n")
    
    all_features = []
    failed_districts = []
    
    for i, (name, west, south, east, north) in enumerate(KERALA_DISTRICTS, 1):
        print(f"[{i}/{len(KERALA_DISTRICTS)}] {name}...", end=" ", flush=True)
        
        query = build_overpass_query(south, west, north, east)
        result = query_overpass(query)
        
        if result is None:
            print("❌ FAILED")
            failed_districts.append(name)
            continue
        
        features = osm_to_geojson(result)
        print(f"{len(features)} features")
        all_features.extend(features)
        
        # Be polite to free Overpass servers
        time.sleep(3)
    
    # Deduplicate by OSM ID
    print(f"\n📊 Total features collected: {len(all_features)}")
    seen_ids = set()
    unique_features = []
    for f in all_features:
        oid = f['properties'].get('osm_id')
        if oid and oid not in seen_ids:
            seen_ids.add(oid)
            unique_features.append(f)
    print(f"   After deduplication: {len(unique_features)}")
    
    if not unique_features:
        print("\n❌ No features collected. Overpass API may be down.")
        return
    
    # Save GeoJSON
    print(f"\n💾 Saving to {OUTPUT_FILE.name}...")
    geojson = {
        "type": "FeatureCollection",
        "features": unique_features
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(geojson, f)
    
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved {size_mb:.1f} MB")
    
    # Statistics by waterway type
    print(f"\n📈 Waterway type breakdown:")
    type_counts = {}
    for f in unique_features:
        props = f['properties']
        wtype = props.get('waterway') or props.get('natural') or props.get('water') or 'unknown'
        type_counts[wtype] = type_counts.get(wtype, 0) + 1
    
    for wtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"   {wtype:<25} {count:>6}")
    
    if failed_districts:
        print(f"\n⚠️ Failed districts: {failed_districts}")
        print(f"   (You can re-run this script to retry just these)")
    
    print("\n" + "="*70)
    print("✅ STEP 7 COMPLETE")
    print("="*70)
    print(f"\n👉 Next: Run Step 8 to recalculate distance-to-water with new data")


if __name__ == "__main__":
    main()