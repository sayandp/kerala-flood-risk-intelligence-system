"""
============================================================
src/02_calculate_distance_to_river.py
============================================================
STEP 2 of dataset rebuild

Creates a raster where each pixel = distance (in meters) to
the nearest river. This is the MISSING FEATURE that will fix
your model — properties close to rivers flood, far ones don't.

INPUT:  data/raw/osm_kerala/kerala_rivers.geojson (11,873 rivers)
        data/raw/copernicus_dem/copernicus_glo30_kerala.tif (template)

OUTPUT: data/processed/distance_to_river_m.tif
        data/processed/distance_to_coast_m.tif
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
import time


# ─── Paths ────────────────────────────────────────────────
RIVERS_PATH = project_root / "data" / "raw" / "osm_kerala" / "kerala_rivers.geojson"
COASTLINE_PATH = project_root / "data" / "raw" / "osm_kerala" / "kerala_coastline.geojson"
DEM_PATH = project_root / "data" / "raw" / "copernicus_dem" / "copernicus_glo30_kerala.tif"

DIST_RIVER_OUTPUT = project_root / "data" / "processed" / "distance_to_river_m.tif"
DIST_COAST_OUTPUT = project_root / "data" / "processed" / "distance_to_coast_m.tif"


def get_template_metadata():
    """Use DEM as template for output rasters (same grid, same CRS)."""
    print("\n📂 Reading DEM as template...")
    with rasterio.open(DEM_PATH) as src:
        meta = {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'pixel_size_deg': abs(src.transform[0])  # degrees per pixel
        }
        print(f"   Template grid: {meta['width']} x {meta['height']}")
        print(f"   CRS: {meta['crs']}")
        print(f"   Pixel size: {meta['pixel_size_deg']:.6f} degrees")
    return meta


def rasterize_geojson_lines(geojson_path, template_meta, name="features"):
    """
    Convert vector lines (rivers/coastline) into a raster mask.
    Returns: numpy array where 1 = line is here, 0 = empty
    """
    print(f"\n🗺️  Rasterizing {name}...")
    
    # Load geojson
    with open(geojson_path, 'r', encoding='utf-8') as f:
        geojson_data = json.load(f)
    
    features = geojson_data.get('features', [])
    print(f"   Loaded {len(features)} {name}")
    
    if not features:
        print(f"   ⚠️ No features found!")
        return None
    
    # Convert each feature to shapely geometry with value=1
    shapes_to_burn = []
    skipped = 0
    
    for feat in features:
        try:
            geom = shape(feat['geometry'])
            shapes_to_burn.append((geom, 1))
        except Exception:
            skipped += 1
    
    if skipped > 0:
        print(f"   Skipped {skipped} invalid geometries")
    
    print(f"   Burning {len(shapes_to_burn)} shapes onto raster grid...")
    
    # Rasterize: lines become 1, everything else 0
    mask = rasterize(
        shapes_to_burn,
        out_shape=(template_meta['height'], template_meta['width']),
        transform=template_meta['transform'],
        fill=0,
        default_value=1,
        dtype='uint8'
    )
    
    river_pixels = (mask == 1).sum()
    total_pixels = mask.size
    coverage_pct = (river_pixels / total_pixels) * 100
    print(f"   ✓ {river_pixels:,} pixels marked as {name} ({coverage_pct:.3f}% of grid)")
    
    return mask


def calculate_distance_in_meters(line_mask, template_meta):
    """
    Compute distance from each pixel to nearest line pixel.
    Returns distance in METERS (not pixels).
    
    This is the magic: scipy's distance_transform_edt is super fast.
    For Kerala (9390 x 16675 = 156M pixels), takes ~30 seconds.
    """
    print(f"\n📏 Calculating distances (this takes ~30 seconds)...")
    start = time.time()
    
    # distance_transform_edt computes Euclidean distance from non-zero pixels
    # to nearest zero pixel — but we want distance FROM zero pixels TO nearest 1
    # So we invert: 0 becomes 1, 1 becomes 0
    inverted = 1 - line_mask.astype(np.int8)
    
    # Compute distance in pixel units
    distance_pixels = distance_transform_edt(inverted)
    
    # Convert pixel distance to METERS
    # At Kerala's latitude (~10°N), 1 degree ≈ 111 km
    # Pixel size in degrees * 111000 = pixel size in meters (latitude)
    # For longitude at 10°N: 1 degree ≈ 109.5 km (cosine effect)
    pixel_size_m = template_meta['pixel_size_deg'] * 111000  # ≈30m for GLO-30
    
    distance_meters = (distance_pixels * pixel_size_m).astype(np.float32)
    
    elapsed = time.time() - start
    print(f"   ✓ Done in {elapsed:.1f} seconds")
    print(f"   Min distance: {distance_meters.min():.1f}m (pixel ON a river)")
    print(f"   Max distance: {distance_meters.max():.1f}m")
    print(f"   Mean: {distance_meters.mean():.1f}m")
    
    return distance_meters


def save_raster(distance_array, template_meta, output_path, description):
    """Save distance array as a GeoTIFF matching the DEM grid."""
    print(f"\n💾 Saving {output_path.name}...")
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=template_meta['height'],
        width=template_meta['width'],
        count=1,
        dtype='float32',
        crs=template_meta['crs'],
        transform=template_meta['transform'],
        compress='lzw',
        nodata=-9999.0
    ) as dst:
        dst.write(distance_array, 1)
        dst.set_band_description(1, description)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✓ Saved ({size_mb:.1f} MB)")


def sanity_check(distance_path, label):
    """Verify the distance raster makes sense by sampling known locations."""
    print(f"\n🧪 Sanity checking {label}...")
    
    test_locations = [
        ("Kuttanad (in flood zone)",  9.4507, 76.4308, "should be NEAR river"),
        ("Aluva (Periyar bank)",     10.1077, 76.3498, "should be NEAR river"),
        ("Munnar (mountain top)",    10.0889, 77.0595, "could be far"),
        ("Edappally (urban)",         9.9893, 76.3060, "moderate"),
    ]
    
    with rasterio.open(distance_path) as src:
        for name, lat, lon, expected in test_locations:
            try:
                row, col = src.index(lon, lat)
                dist = src.read(1)[row, col]
                
                # Format nicely
                if dist < 100:
                    dist_str = f"{dist:.0f} m"
                elif dist < 1000:
                    dist_str = f"{dist:.0f} m"
                else:
                    dist_str = f"{dist/1000:.2f} km"
                
                print(f"   {name:<32} {dist_str:>10}  ({expected})")
            except Exception as e:
                print(f"   {name:<32} ERROR: {e}")


def main():
    print("\n" + "="*60)
    print("STEP 2: CALCULATE DISTANCE TO RIVERS & COAST")
    print("="*60)
    
    # Verify inputs exist
    if not RIVERS_PATH.exists():
        print(f"❌ Rivers file missing: {RIVERS_PATH}")
        print("   Run Step 1 first: python src/01_download_rivers.py")
        return
    
    if not DEM_PATH.exists():
        print(f"❌ DEM file missing: {DEM_PATH}")
        return
    
    # Get template grid from DEM
    template = get_template_metadata()
    
    # ─── Process Rivers ───────────────────────────────────
    print("\n" + "─"*60)
    print(" RIVERS")
    print("─"*60)
    
    river_mask = rasterize_geojson_lines(RIVERS_PATH, template, "rivers")
    if river_mask is not None:
        river_distance = calculate_distance_in_meters(river_mask, template)
        save_raster(river_distance, template, DIST_RIVER_OUTPUT, "Distance to nearest river (meters)")
        sanity_check(DIST_RIVER_OUTPUT, "river distances")
    
    # ─── Process Coast ────────────────────────────────────
    if COASTLINE_PATH.exists():
        print("\n" + "─"*60)
        print(" COASTLINE")
        print("─"*60)
        
        coast_mask = rasterize_geojson_lines(COASTLINE_PATH, template, "coastline")
        if coast_mask is not None:
            coast_distance = calculate_distance_in_meters(coast_mask, template)
            save_raster(coast_distance, template, DIST_COAST_OUTPUT, "Distance to coast (meters)")
            sanity_check(DIST_COAST_OUTPUT, "coast distances")
    
    # ─── Final Summary ────────────────────────────────────
    print("\n" + "="*60)
    print("✅ STEP 2 COMPLETE")
    print("="*60)
    print(f"\nOutput files:")
    if DIST_RIVER_OUTPUT.exists():
        print(f"  ✓ {DIST_RIVER_OUTPUT}")
    if DIST_COAST_OUTPUT.exists():
        print(f"  ✓ {DIST_COAST_OUTPUT}")
    print(f"\n👉 Next: Step 3 — calculate slope from DEM")


if __name__ == "__main__":
    main()