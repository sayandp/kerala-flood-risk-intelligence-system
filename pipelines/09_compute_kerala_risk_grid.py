"""
============================================================
pipelines/09_compute_kerala_risk_grid.py
============================================================
Pre-computes flood risk for ~3500 points across Kerala.
Output is used by the frontend to render a state-wide heatmap.

OUTPUT: data/processed/kerala_risk_grid.json
        Format: {"metadata": {...}, "points": [{"lat": 10.0, "lng": 76.5, "risk": 0.65}, ...]}

Runtime: ~3-5 minutes

UPDATED: Uses smart paddy extraction with elevation filter (buf=50 + max_elev=50m)
         to match backend/main.py logic. Old version used buf=5 (only 50m radius)
         which silently failed for almost every grid point.
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
import time

from src.hybrid_risk_scorer import apply_business_rules


# ─── Configuration ────────────────────────────────────────
MODEL_PATH      = project_root / "models"     / "flood_rf_model.pkl"
DEM_PATH        = project_root / "data" / "raw" / "copernicus_dem"   / "copernicus_glo30_kerala.tif"
FLOW_PATH       = project_root / "data" / "processed"                / "kerala_flow_accumulation_km2.tif"
PADDY_PATH      = project_root / "data" / "raw" / "lulc_2020"        / "historical_paddy_kerala.tif"
RIVER_DIST_PATH = project_root / "data" / "processed"                / "distance_to_river_m.tif"
SLOPE_PATH      = project_root / "data" / "processed"                / "slope_degrees.tif"

OUTPUT_PATH = project_root / "data" / "processed" / "kerala_risk_grid.json"

FEATURE_COLS = [
    'elevation_m',
    'upstream_catchment_km2',
    'is_historical_paddy',
    'distance_to_river_m',
    'slope_degrees',
]

KERALA_BOUNDS = {
    'lat_min':  8.0,
    'lat_max': 12.8,
    'lng_min': 74.5,
    'lng_max': 77.5,
}

# Grid resolution:
# 0.04° ≈ 4.4 km spacing → ~5,000 points
# 0.03° ≈ 3.3 km spacing → ~9,000 points (recommended)
# 0.02° ≈ 2.2 km spacing → ~20,000 points (slow)
GRID_SPACING_DEG = 0.03

# Elevation filter for paddy — matches backend/main.py
PADDY_MAX_ELEVATION_M = 50.0


# ─── Helper functions ─────────────────────────────────────
def extract_with_buffer(src, lat, lon, buffer=3, stat='median'):
    """Extract feature with smart buffer sampling."""
    try:
        r, c = src.index(lon, lat)
        col_off = max(0, c - buffer)
        row_off = max(0, r - buffer)
        col_end = min(src.width,  c + buffer + 1)
        row_end = min(src.height, r + buffer + 1)
        win_w   = col_end - col_off
        win_h   = row_end - row_off

        if win_w <= 0 or win_h <= 0:
            return None

        window = Window(col_off, row_off, win_w, win_h)
        data   = src.read(1, window=window)

        nodata = src.nodata
        if nodata is not None:
            valid = data[data != nodata]
        elif np.issubdtype(data.dtype, np.floating):
            valid = data[~np.isnan(data)]
        else:
            valid = data.flatten()

        if len(valid) == 0:
            return None

        if stat == 'median': return float(np.median(valid))
        if stat == 'max':    return float(np.max(valid))
        if stat == 'min':    return float(np.min(valid))
        if stat == 'any':    return int(np.any(valid > 0))
        return None
    except Exception:
        return None


def extract_paddy_smart(paddy_src, dem_src, lat, lon, buf=50, max_elev=PADDY_MAX_ELEVATION_M):
    """
    Detect paddy land within 500m, BUT only if local elevation is below 50m.

    Why the elevation filter:
      Dynamic World's 'crops' class sometimes mislabels Western Ghats tea
      plantations and terraced fields as paddy. Kerala paddy land is by
      definition low-lying — so we suppress paddy detection above 50m.
    """
    try:
        r_d, c_d  = dem_src.index(lon, lat)
        elev_data = dem_src.read(1, window=Window(c_d, r_d, 1, 1))
        if elev_data.size == 0:
            return 0
        elev = float(elev_data[0, 0])

        if elev > max_elev:
            return 0

        r_p, c_p = paddy_src.index(lon, lat)
        col_off  = max(0, c_p - buf)
        row_off  = max(0, r_p - buf)
        col_end  = min(paddy_src.width,  c_p + buf + 1)
        row_end  = min(paddy_src.height, r_p + buf + 1)
        win_w    = col_end - col_off
        win_h    = row_end - row_off

        if win_w <= 0 or win_h <= 0:
            return 0

        paddy_data = paddy_src.read(1, window=Window(col_off, row_off, win_w, win_h))
        return int(np.any(paddy_data == 1))
    except Exception:
        return 0


def is_in_kerala(lat, lng):
    return (KERALA_BOUNDS['lat_min'] <= lat <= KERALA_BOUNDS['lat_max'] and
            KERALA_BOUNDS['lng_min'] <= lng <= KERALA_BOUNDS['lng_max'])


def main():
    print("\n" + "="*70)
    print("STEP 9: COMPUTE KERALA RISK GRID (for heatmap)")
    print("Smart paddy extraction with elevation filter ENABLED")
    print("="*70)

    print("\n[1/4] Loading model...")
    model = joblib.load(MODEL_PATH)
    print(f"   model loaded")

    print("\n[2/4] Opening raster files...")
    src_dem   = rasterio.open(DEM_PATH)
    src_flow  = rasterio.open(FLOW_PATH)
    src_paddy = rasterio.open(PADDY_PATH)
    src_river = rasterio.open(RIVER_DIST_PATH)
    src_slope = rasterio.open(SLOPE_PATH)
    print(f"   all rasters open")

    print(f"\n[3/4] Generating grid points...")
    lats = np.arange(KERALA_BOUNDS['lat_min'], KERALA_BOUNDS['lat_max'], GRID_SPACING_DEG)
    lngs = np.arange(KERALA_BOUNDS['lng_min'], KERALA_BOUNDS['lng_max'], GRID_SPACING_DEG)
    print(f"   Grid: {len(lats)} x {len(lngs)} = {len(lats) * len(lngs):,} points")

    print(f"\n[4/4] Computing risk scores...")
    grid_results    = []
    skipped_no_data = 0
    skipped_water   = 0

    total = len(lats) * len(lngs)
    pbar  = tqdm(total=total, unit='points')
    start_time = time.time()

    for lat in lats:
        for lng in lngs:
            pbar.update(1)

            elevation = extract_with_buffer(src_dem, lat, lng, 3, 'median')

            if elevation is None or elevation < -10:
                skipped_no_data += 1
                continue

            if elevation < 0:
                skipped_water += 1
                continue

            catchment  = extract_with_buffer(src_flow,  lat, lng, 3, 'max')
            river_dist = extract_with_buffer(src_river, lat, lng, 3, 'min')
            slope      = extract_with_buffer(src_slope, lat, lng, 3, 'median')

            # ── SMART PADDY EXTRACTION ─────────────────────
            paddy = extract_paddy_smart(
                src_paddy, src_dem, lat, lng,
                buf=50, max_elev=PADDY_MAX_ELEVATION_M
            )

            if catchment  is None: catchment  = 0.5
            if river_dist is None: river_dist = 1000
            if slope      is None: slope      = 5.0

            features = {
                'elevation_m':            elevation,
                'upstream_catchment_km2': catchment,
                'is_historical_paddy':    int(paddy),
                'distance_to_river_m':    river_dist,
                'slope_degrees':          slope,
            }

            try:
                feature_df = pd.DataFrame([[
                    features['elevation_m'],
                    features['upstream_catchment_km2'],
                    features['is_historical_paddy'],
                    features['distance_to_river_m'],
                    features['slope_degrees'],
                ]], columns=FEATURE_COLS)

                ml_prob     = float(model.predict_proba(feature_df)[0][1])
                rule_result = apply_business_rules(features, ml_prob)
                final_prob  = rule_result['final_probability']

                grid_results.append({
                    'lat':  round(float(lat), 4),
                    'lng':  round(float(lng), 4),
                    'risk': round(final_prob, 3),
                })
            except Exception:
                skipped_no_data += 1
                continue

    pbar.close()

    elapsed = time.time() - start_time
    print(f"\n   computed in {elapsed:.1f}s")
    print(f"   valid points:      {len(grid_results):,}")
    print(f"   skipped (water):   {skipped_water:,}")
    print(f"   skipped (no data): {skipped_no_data:,}")

    src_dem.close()
    src_flow.close()
    src_paddy.close()
    src_river.close()
    src_slope.close()

    if grid_results:
        risks = [p['risk'] for p in grid_results]
        print(f"\nRisk distribution:")
        print(f"   Min:    {min(risks)*100:.1f}%")
        print(f"   Max:    {max(risks)*100:.1f}%")
        print(f"   Mean:   {np.mean(risks)*100:.1f}%")
        print(f"   Median: {np.median(risks)*100:.1f}%")

        very_low = sum(1 for r in risks if r < 0.20)
        low      = sum(1 for r in risks if 0.20 <= r < 0.40)
        moderate = sum(1 for r in risks if 0.40 <= r < 0.60)
        high     = sum(1 for r in risks if 0.60 <= r < 0.80)
        critical = sum(1 for r in risks if r >= 0.80)

        total_valid = len(risks)
        print(f"\nRisk bands:")
        print(f"   VERY LOW  (<20%):   {very_low:>5} ({very_low/total_valid*100:5.1f}%)")
        print(f"   LOW       (20-40%): {low:>5} ({low/total_valid*100:5.1f}%)")
        print(f"   MODERATE  (40-60%): {moderate:>5} ({moderate/total_valid*100:5.1f}%)")
        print(f"   HIGH      (60-80%): {high:>5} ({high/total_valid*100:5.1f}%)")
        print(f"   CRITICAL  (>=80%):  {critical:>5} ({critical/total_valid*100:5.1f}%)")

    print(f"\nSaving grid...")
    output_data = {
        'metadata': {
            'total_points':           len(grid_results),
            'grid_spacing_deg':       GRID_SPACING_DEG,
            'grid_spacing_km_approx': round(GRID_SPACING_DEG * 111, 1),
            'model_accuracy':         0.9196,
            'paddy_logic':            'smart_elevation_filter_v2',
            'paddy_max_elev_m':       PADDY_MAX_ELEVATION_M,
            'description':            'Kerala flood risk grid for heatmap visualization',
        },
        'points': grid_results,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output_data, f, separators=(',', ':'))

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"   saved {OUTPUT_PATH.name} ({size_kb:.1f} KB)")

    print("\n" + "="*70)
    print("STEP 9 COMPLETE")
    print("="*70)
    


if __name__ == "__main__":
    main()