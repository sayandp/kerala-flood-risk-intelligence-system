"""
============================================================
src/03_calculate_slope.py
============================================================
STEP 3 of dataset rebuild

Calculates the slope (steepness) of the land at every Kerala 
location from the Copernicus DEM.

WHY THIS MATTERS:
- Flat land (slope ≈ 0°) → water pools → high flood risk
- Steep slopes (slope > 15°) → water flows away → low flood risk

INPUT:  data/raw/copernicus_dem/copernicus_glo30_kerala.tif
OUTPUT: data/processed/slope_degrees.tif
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import rasterio
from rasterio.windows import Window
import time


# ─── Paths ────────────────────────────────────────────────
DEM_PATH = project_root / "data" / "raw" / "copernicus_dem" / "copernicus_glo30_kerala.tif"
SLOPE_OUTPUT = project_root / "data" / "processed" / "slope_degrees.tif"


def calculate_slope(dem_array, pixel_size_m):
    """
    Calculate slope in degrees from elevation data.
    
    Method: Horn's algorithm (the same one ArcGIS/QGIS uses)
    Computes gradient using 3x3 neighbor windows.
    
    Args:
        dem_array: 2D numpy array of elevations
        pixel_size_m: size of each pixel in meters (~30m for GLO-30)
    
    Returns:
        2D array of slope values in DEGREES
    """
    # Compute partial derivatives using numpy gradient (faster than loops)
    # axis=0 = north-south direction (rows)
    # axis=1 = east-west direction (cols)
    dz_dy, dz_dx = np.gradient(dem_array, pixel_size_m)
    
    # Slope magnitude = sqrt(dx² + dy²)
    slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    
    # Convert to degrees
    slope_degrees = np.degrees(slope_radians).astype(np.float32)
    
    return slope_degrees


def process_dem_in_chunks():
    """
    Process the DEM in chunks to avoid memory issues.
    Kerala DEM is 9390 x 16675 = 156M pixels = ~600MB in float32.
    
    Strategy: Read, process, and write in horizontal strips.
    """
    print(f"\n📂 Opening DEM: {DEM_PATH.name}")
    
    with rasterio.open(DEM_PATH) as src:
        print(f"   Dimensions: {src.width} x {src.height}")
        print(f"   CRS: {src.crs}")
        
        # Calculate pixel size in meters (Kerala latitude ~10°N)
        pixel_size_deg = abs(src.transform[0])
        pixel_size_m = pixel_size_deg * 111000  # ~30m for GLO-30
        print(f"   Pixel size: {pixel_size_m:.1f}m")
        
        # Read entire DEM into memory (it's only ~600MB)
        print(f"\n📥 Loading DEM into memory...")
        start = time.time()
        dem = src.read(1).astype(np.float32)
        print(f"   ✓ Loaded in {time.time()-start:.1f}s")
        
        # Handle NaN/nodata values
        print(f"\n🧹 Handling missing values...")
        if src.nodata is not None:
            dem[dem == src.nodata] = 0
        dem[np.isnan(dem)] = 0
        # Cap unreasonable values
        dem[dem < -10] = 0
        dem[dem > 3000] = 3000
        print(f"   Min elevation: {dem.min():.0f}m")
        print(f"   Max elevation: {dem.max():.0f}m")
        
        # Calculate slope
        print(f"\n📏 Calculating slope (this takes ~30 seconds)...")
        start = time.time()
        slope = calculate_slope(dem, pixel_size_m)
        elapsed = time.time() - start
        print(f"   ✓ Done in {elapsed:.1f}s")
        
        # Statistics
        print(f"\n📊 Slope statistics:")
        print(f"   Min:    {slope.min():.2f}° (perfectly flat)")
        print(f"   Max:    {slope.max():.2f}° (cliffs/mountains)")
        print(f"   Mean:   {slope.mean():.2f}°")
        print(f"   Median: {np.median(slope):.2f}°")
        
        # Distribution check
        flat = (slope < 1).sum()
        gentle = ((slope >= 1) & (slope < 5)).sum()
        moderate = ((slope >= 5) & (slope < 15)).sum()
        steep = (slope >= 15).sum()
        total = slope.size
        
        print(f"\n   Distribution:")
        print(f"   Flat       (< 1°):   {flat/total*100:5.1f}%  ({flat:,} pixels)")
        print(f"   Gentle  (1-5°):      {gentle/total*100:5.1f}%  ({gentle:,} pixels)")
        print(f"   Moderate(5-15°):     {moderate/total*100:5.1f}%  ({moderate:,} pixels)")
        print(f"   Steep    (> 15°):    {steep/total*100:5.1f}%  ({steep:,} pixels)")
        
        # Save with same metadata as DEM
        meta = src.meta.copy()
        meta.update({
            'dtype': 'float32',
            'compress': 'lzw',
            'nodata': -9999.0
        })
        
        print(f"\n💾 Saving slope raster...")
        with rasterio.open(SLOPE_OUTPUT, 'w', **meta) as dst:
            dst.write(slope, 1)
            dst.set_band_description(1, "Slope in degrees")
        
        size_mb = SLOPE_OUTPUT.stat().st_size / (1024 * 1024)
        print(f"   ✓ Saved ({size_mb:.1f} MB)")
        
        return slope


def sanity_check():
    """Verify slope makes sense at known locations."""
    print(f"\n🧪 Sanity checking slope values...")
    
    test_locations = [
        ("Kuttanad (flat backwaters)",  9.4507, 76.4308, "should be FLAT < 2°"),
        ("Aluva (river plain)",        10.1077, 76.3498, "should be FLAT < 5°"),
        ("Munnar (mountain ridge)",    10.0889, 77.0595, "should be STEEP > 10°"),
        ("Vagamon (hills)",             9.6783, 76.9067, "should be MODERATE 5-15°"),
        ("Wayanad-Kalpetta (hills)",   11.6094, 76.0838, "should be MODERATE-STEEP"),
        ("Kochi (coastal flat)",        9.9312, 76.2673, "should be FLAT < 3°"),
    ]
    
    with rasterio.open(SLOPE_OUTPUT) as src:
        for name, lat, lon, expected in test_locations:
            try:
                row, col = src.index(lon, lat)
                slope_val = src.read(1)[row, col]
                
                # Classify
                if slope_val < 2:
                    label = "FLAT     "
                elif slope_val < 5:
                    label = "GENTLE   "
                elif slope_val < 15:
                    label = "MODERATE "
                else:
                    label = "STEEP    "
                
                print(f"   {name:<32} {slope_val:5.2f}°  {label}  ({expected})")
            except Exception as e:
                print(f"   {name:<32} ERROR: {e}")


def main():
    print("\n" + "="*60)
    print("STEP 3: CALCULATE SLOPE FROM DEM")
    print("="*60)
    
    if not DEM_PATH.exists():
        print(f"❌ DEM file missing: {DEM_PATH}")
        return
    
    slope = process_dem_in_chunks()
    sanity_check()
    
    print("\n" + "="*60)
    print("✅ STEP 3 COMPLETE")
    print("="*60)
    print(f"\n👉 Next: Step 4 — build new training dataset with all features")


if __name__ == "__main__":
    main()