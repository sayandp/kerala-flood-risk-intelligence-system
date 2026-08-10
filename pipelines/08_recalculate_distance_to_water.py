"""
src/08_recalculate_distance_to_water.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
import time
import shutil

WATERWAYS_PATH = project_root / "data" / "raw" / "osm_kerala" / "kerala_waterways_osm.geojson"
DEM_PATH = project_root / "data" / "raw" / "copernicus_dem" / "copernicus_glo30_kerala.tif"
NEW_DISTANCE_OUTPUT = project_root / "data" / "processed" / "distance_to_water_m.tif"
OLD_RIVER_DISTANCE = project_root / "data" / "processed" / "distance_to_river_m.tif"


def get_template_metadata():
    print("\nReading DEM as template...")
    with rasterio.open(DEM_PATH) as src:
        return {
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height,
            'pixel_size_deg': abs(src.transform[0])
        }


def rasterize_waterways(geojson_path, template_meta):
    print("\nLoading waterways geojson (80MB, may take a minute)...")
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    features = data.get('features', [])
    print(f"   Loaded {len(features):,} features")
    
    print("\nProcessing geometries...")
    shapes_to_burn = []
    skipped = 0
    line_count = 0
    poly_count = 0
    
    for i, feat in enumerate(features):
        if i % 20000 == 0 and i > 0:
            print(f"   Processed {i:,} / {len(features):,}...")
        try:
            geom_dict = feat['geometry']
            geom_type = geom_dict.get('type')
            if geom_type in ('LineString', 'Polygon', 'MultiPolygon'):
                geom = shape(geom_dict)
                shapes_to_burn.append((geom, 1))
                if geom_type == 'LineString':
                    line_count += 1
                else:
                    poly_count += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
    
    print(f"   Lines: {line_count:,}, Polygons: {poly_count:,}, Skipped: {skipped:,}")
    
    print(f"\nRasterizing onto {template_meta['width']}x{template_meta['height']} grid...")
    print("   (slow step, takes 1-3 minutes)")
    start = time.time()
    
    mask = rasterize(
        shapes_to_burn,
        out_shape=(template_meta['height'], template_meta['width']),
        transform=template_meta['transform'],
        fill=0,
        default_value=1,
        dtype='uint8'
    )
    
    print(f"   Done in {time.time()-start:.1f}s")
    water_pixels = (mask == 1).sum()
    print(f"   Water coverage: {water_pixels:,} pixels ({water_pixels/mask.size*100:.2f}%)")
    return mask


def calculate_distance_meters(water_mask, template_meta):
    print("\nCalculating distance transform...")
    start = time.time()
    inverted = 1 - water_mask.astype(np.int8)
    distance_pixels = distance_transform_edt(inverted)
    pixel_size_m = template_meta['pixel_size_deg'] * 111000
    distance_meters = (distance_pixels * pixel_size_m).astype(np.float32)
    print(f"   Done in {time.time()-start:.1f}s")
    print(f"   Min: {distance_meters.min():.1f}m, Max: {distance_meters.max():.1f}m")
    print(f"   Mean: {distance_meters.mean():.1f}m, Median: {np.median(distance_meters):.1f}m")
    return distance_meters


def save_raster(distance_array, template_meta, output_path):
    print(f"\nSaving {output_path.name}...")
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=template_meta['height'], width=template_meta['width'],
        count=1, dtype='float32',
        crs=template_meta['crs'], transform=template_meta['transform'],
        compress='lzw', nodata=-9999.0
    ) as dst:
        dst.write(distance_array, 1)
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Saved ({size_mb:.1f} MB)")


def sanity_check():
    print("\nSanity checking distances...")
    test_locations = [
        ("Kuttanad",      9.4507, 76.4308, "should be ~0m"),
        ("Aluva",        10.1077, 76.3498, "should be ~0m"),
        ("Vembanad isle", 9.6500, 76.4200, "should be <100m"),
        ("Munnar",       10.0889, 77.0595, "could be far"),
        ("Kochi",         9.9312, 76.2673, "should be near water"),
        ("Edappally",     9.9893, 76.3060, "urban, near canals"),
    ]
    
    with rasterio.open(NEW_DISTANCE_OUTPUT) as src:
        for name, lat, lon, expected in test_locations:
            try:
                row, col = src.index(lon, lat)
                dist = src.read(1)[row, col]
                if dist < 1000:
                    dist_str = f"{dist:.0f}m"
                else:
                    dist_str = f"{dist/1000:.2f}km"
                print(f"   {name:<18} {dist_str:>10}  ({expected})")
            except Exception as e:
                print(f"   {name:<18} ERROR: {e}")


def main():
    print("\n" + "="*70)
    print("STEP 8: RECALCULATE DISTANCE-TO-WATER (using OSM)")
    print("="*70)
    
    if not WATERWAYS_PATH.exists():
        print(f"Waterways file missing: {WATERWAYS_PATH}")
        return
    if not DEM_PATH.exists():
        print("DEM missing")
        return
    
    template = get_template_metadata()
    water_mask = rasterize_waterways(WATERWAYS_PATH, template)
    distances = calculate_distance_meters(water_mask, template)
    save_raster(distances, template, NEW_DISTANCE_OUTPUT)
    
    if OLD_RIVER_DISTANCE.exists():
        backup_path = OLD_RIVER_DISTANCE.with_suffix('.tif.OLD_RIVER_BACKUP')
        try:
            OLD_RIVER_DISTANCE.rename(backup_path)
            print(f"\nBacked up old river file")
        except Exception:
            pass
    
    print("\nCreating compatibility link...")
    shutil.copy(NEW_DISTANCE_OUTPUT, project_root / "data" / "processed" / "distance_to_river_m.tif")
    print("   distance_to_river_m.tif now uses comprehensive water data")
    
    sanity_check()
    
    print("\n" + "="*70)
    print("STEP 8 COMPLETE")
    print("="*70)
    print("\nRestart backend: python main.py")


if __name__ == "__main__":
    main()