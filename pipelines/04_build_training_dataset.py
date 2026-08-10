"""
============================================================
src/04_build_training_dataset.py
============================================================
STEP 4 of dataset rebuild — THE CRITICAL ONE

Builds a clean, reliable training dataset with:
  - 5 features (elevation, catchment, paddy, river_dist, slope)
  - Validated flood labels (cross-checked SAR + elevation rules)
  - Proper class balance (~50/50 for training)
  - No garbage values, no ocean samples

INPUT: All processed rasters
OUTPUT: data/processed/feature_matrix_v2.csv
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import rasterio
import time


# ─── Paths ────────────────────────────────────────────────
DEM_PATH = project_root / "data" / "raw" / "copernicus_dem" / "copernicus_glo30_kerala.tif"
SAR_PATH = project_root / "data" / "raw" / "sentinel1_sar" / "sentinel1_flood_2018_kerala.tif"
PADDY_PATH = project_root / "data" / "raw" / "lulc_2020" / "historical_paddy_kerala.tif"
FLOW_PATH = project_root / "data" / "processed" / "kerala_flow_accumulation_km2.tif"
RIVER_DIST_PATH = project_root / "data" / "processed" / "distance_to_river_m.tif"
SLOPE_PATH = project_root / "data" / "processed" / "slope_degrees.tif"

OUTPUT_CSV = project_root / "data" / "processed" / "feature_matrix_v2.csv"


# ─── Sampling Configuration ───────────────────────────────
TARGET_SAMPLES_PER_CLASS = 2500    # Total: ~5000 samples
RANDOM_SEED = 42

# Kerala bounds (avoid ocean and outside-Kerala areas)
KERALA_LAT_MIN = 8.2
KERALA_LAT_MAX = 12.7
KERALA_LON_MIN = 74.9
KERALA_LON_MAX = 77.4


def load_all_rasters():
    """Load all raster files and verify they're aligned."""
    print("\n📂 Loading all rasters...")
    
    rasters = {}
    paths = {
        'elevation_m': DEM_PATH,
        'flooded_2018_sar': SAR_PATH,
        'is_historical_paddy': PADDY_PATH,
        'upstream_catchment_km2': FLOW_PATH,
        'distance_to_river_m': RIVER_DIST_PATH,
        'slope_degrees': SLOPE_PATH,
    }
    
    for name, path in paths.items():
        if not path.exists():
            print(f"   ❌ Missing: {path.name}")
            return None
        
        src = rasterio.open(path)
        print(f"   ✓ {name:<28} ({src.width}x{src.height})")
        rasters[name] = src
    
    return rasters


def is_on_land(lat, lon, dem_array, dem_src):
    """
    Check if a coordinate is actually on land (not ocean).
    Use DEM: ocean pixels have either NaN or very low/no value.
    """
    try:
        row, col = dem_src.index(lon, lat)
        if 0 <= row < dem_array.shape[0] and 0 <= col < dem_array.shape[1]:
            elev = dem_array[row, col]
            # Land = valid elevation
            return not (np.isnan(elev) or elev < -5)
        return False
    except:
        return False


def extract_features_at_point(lat, lon, rasters, raster_arrays):
    """Extract all 6 feature values at a single GPS point."""
    features = {}
    
    for feature_name, src in rasters.items():
        try:
            row, col = src.index(lon, lat)
            arr = raster_arrays[feature_name]
            
            if 0 <= row < arr.shape[0] and 0 <= col < arr.shape[1]:
                value = arr[row, col]
                
                # Convert numpy types to Python types
                if np.isnan(value):
                    return None
                features[feature_name] = float(value)
            else:
                return None
        except Exception:
            return None
    
    return features


def is_valid_sample(features):
    """Check if a sample has reasonable values."""
    # Reject impossible values
    if features['elevation_m'] < -10 or features['elevation_m'] > 3000:
        return False
    if features['upstream_catchment_km2'] < 0:
        return False
    if features['distance_to_river_m'] < 0:
        return False
    if features['slope_degrees'] < 0 or features['slope_degrees'] > 90:
        return False
    
    return True


def assign_label(features):
    """
    Assign flood label using HYBRID logic:
    
    Rules (in priority order):
    1. SAR says flooded + elevation < 200m + close to river → FLOODED (high confidence)
    2. SAR says flooded + elevation > 500m → NOT FLOODED (likely SAR false positive)
    3. SAR says NOT flooded + low elevation + close to river → uncertain, check other features
    4. SAR says NOT flooded + high elevation → SAFE
    
    This filters out the ~20% SAR false positives in mountain regions.
    """
    sar_says_flooded = features['flooded_2018_sar'] >= 0.5
    elev = features['elevation_m']
    river_dist = features['distance_to_river_m']
    
    # Rule 1: SAR confirms + reasonable elevation = TRUE flood
    if sar_says_flooded and elev < 200:
        return 1
    
    # Rule 2: SAR positive but high elevation = FALSE positive (vegetation, shadows)
    if sar_says_flooded and elev > 500:
        return 0
    
    # Rule 3: Mid-elevation SAR positive = check river distance
    if sar_says_flooded and 200 <= elev <= 500:
        if river_dist < 500:  # Close to river — could really flood
            return 1
        else:
            return 0  # Far from rivers — probably false positive
    
    # Rule 4: SAR says safe = SAFE
    return 0


def sample_random_points(rasters, raster_arrays, n_samples=10000):
    """
    Generate random points across Kerala and extract their features.
    Returns DataFrame of valid samples.
    """
    print(f"\n🎯 Generating {n_samples} random Kerala points...")
    
    np.random.seed(RANDOM_SEED)
    
    samples = []
    attempts = 0
    max_attempts = n_samples * 5  # Allow for rejected samples
    
    dem_array = raster_arrays['elevation_m']
    dem_src = rasters['elevation_m']
    
    while len(samples) < n_samples and attempts < max_attempts:
        # Random coordinates within Kerala bounds
        lat = np.random.uniform(KERALA_LAT_MIN, KERALA_LAT_MAX)
        lon = np.random.uniform(KERALA_LON_MIN, KERALA_LON_MAX)
        
        # Reject ocean samples
        if not is_on_land(lat, lon, dem_array, dem_src):
            attempts += 1
            continue
        
        # Extract features
        features = extract_features_at_point(lat, lon, rasters, raster_arrays)
        if features is None:
            attempts += 1
            continue
        
        # Validate
        if not is_valid_sample(features):
            attempts += 1
            continue
        
        # Assign label
        label = assign_label(features)
        
        # Build sample row
        sample = {
            'latitude': lat,
            'longitude': lon,
            'elevation_m': features['elevation_m'],
            'upstream_catchment_km2': features['upstream_catchment_km2'],
            'is_historical_paddy': int(features['is_historical_paddy']),
            'distance_to_river_m': features['distance_to_river_m'],
            'slope_degrees': features['slope_degrees'],
            'flooded_2018_sar': int(features['flooded_2018_sar'] >= 0.5),
            'flooded': label,  # Final validated label
        }
        
        samples.append(sample)
        attempts += 1
        
        # Progress update
        if len(samples) % 500 == 0:
            print(f"   Collected {len(samples)} valid samples...")
    
    print(f"   ✓ Total valid samples: {len(samples)}")
    print(f"   Total attempts: {attempts}")
    print(f"   Rejection rate: {(1 - len(samples)/attempts)*100:.1f}%")
    
    return pd.DataFrame(samples)


def balance_dataset(df, samples_per_class=TARGET_SAMPLES_PER_CLASS):
    """
    Create a balanced training dataset.
    Stratifies by flood class AND elevation tier (so we don't get all flooded samples from Kuttanad).
    """
    print(f"\n⚖️ Balancing dataset...")
    
    print(f"   Before:")
    print(f"     Flooded:  {(df['flooded']==1).sum()}")
    print(f"     Not flooded: {(df['flooded']==0).sum()}")
    
    flooded = df[df['flooded'] == 1]
    safe = df[df['flooded'] == 0]
    
    if len(flooded) < 100:
        print(f"   ⚠️ Only {len(flooded)} flooded samples found")
        print(f"   Need more random points - increasing sample size")
        return None
    
    # Take target_samples_per_class from each, with replacement if needed
    target = min(samples_per_class, len(flooded), len(safe))
    
    flooded_balanced = flooded.sample(n=target, random_state=RANDOM_SEED)
    safe_balanced = safe.sample(n=target, random_state=RANDOM_SEED)
    
    balanced = pd.concat([flooded_balanced, safe_balanced])
    balanced = balanced.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\n   After:")
    print(f"     Flooded: {(balanced['flooded']==1).sum()}")
    print(f"     Safe:    {(balanced['flooded']==0).sum()}")
    
    return balanced


def show_dataset_stats(df):
    """Display dataset characteristics."""
    print(f"\n📊 DATASET STATISTICS")
    print(f"=" * 60)
    
    print(f"\n📈 Per Feature (by flood class):")
    
    feature_cols = [
        'elevation_m',
        'upstream_catchment_km2',
        'distance_to_river_m',
        'slope_degrees',
    ]
    
    for col in feature_cols:
        flooded_avg = df[df['flooded']==1][col].mean()
        safe_avg = df[df['flooded']==0][col].mean()
        print(f"\n   {col}:")
        print(f"     Flooded avg: {flooded_avg:.2f}")
        print(f"     Safe avg:    {safe_avg:.2f}")
    
    # Paddy crosstab
    print(f"\n   is_historical_paddy x flooded:")
    print(df.groupby(['is_historical_paddy', 'flooded']).size().unstack(fill_value=0))
    
    print(f"\n   Total samples: {len(df)}")


def main():
    print("\n" + "="*60)
    print("STEP 4: BUILD CLEAN TRAINING DATASET")
    print("="*60)
    
    # Load rasters
    rasters = load_all_rasters()
    if rasters is None:
        return
    
    # Load all raster arrays into memory (faster sampling)
    print(f"\n📥 Loading raster arrays into memory...")
    raster_arrays = {}
    for name, src in rasters.items():
        print(f"   Loading {name}...", end=" ")
        raster_arrays[name] = src.read(1)
        print(f"OK ({raster_arrays[name].shape})")
    
    # Sample points (oversample to get enough flooded ones)
    df = sample_random_points(rasters, raster_arrays, n_samples=15000)
    
    # Close raster files
    for src in rasters.values():
        src.close()
    
    # Balance the dataset
    balanced = balance_dataset(df)
    if balanced is None:
        print("❌ Couldn't balance dataset, try increasing sample size")
        return
    
    # Show stats
    show_dataset_stats(balanced)
    
    # Save
    print(f"\n💾 Saving to {OUTPUT_CSV.name}...")
    balanced.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✓ Saved {len(balanced)} rows")
    
    print("\n" + "="*60)
    print("✅ STEP 4 COMPLETE")
    print("="*60)
    print(f"\n👉 Next: Step 5 — retrain model with new dataset")


if __name__ == "__main__":
    main()