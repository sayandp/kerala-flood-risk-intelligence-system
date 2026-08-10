"""
============================================================
src/06_smart_feature_extractor.py
Fix the buggy feature extraction by using buffer zones
============================================================

PROBLEM:
A single pixel often doesn't represent the property realistically:
  - Catchment: only river pixels have values, you sample 50m off
    the river and get 0.00 even though there's a river nearby
  - Slope: rough DEM noise gives weird values for individual pixels  
  - Idukki coordinates fall in the reservoir water itself

SOLUTION:
For each feature, sample a buffer zone (5x5 pixels = ~150m radius)
and use the most realistic statistic:
  - Elevation: median (robust to noise)
  - Catchment: maximum (capture nearby drainage)
  - Distance to river: minimum (closest river point in buffer)
  - Slope: median (robust to noise)
  - Paddy: any (binary OR across buffer)
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import rasterio


def extract_with_buffer(raster_src, lat, lon, buffer_pixels=5, statistic='median'):
    """
    Sample a buffer around the GPS point.
    
    Args:
        raster_src: Open rasterio dataset
        lat, lon: GPS coordinates  
        buffer_pixels: How many pixels around (5 = ~150m radius for 30m DEM)
        statistic: 'median', 'mean', 'min', 'max', 'any'
    
    Returns:
        Single value summarizing the buffer area
    """
    try:
        # Get center pixel
        center_row, center_col = raster_src.index(lon, lat)
        
        # Define buffer window
        row_start = max(0, center_row - buffer_pixels)
        row_end = min(raster_src.height, center_row + buffer_pixels + 1)
        col_start = max(0, center_col - buffer_pixels)
        col_end = min(raster_src.width, center_col + buffer_pixels + 1)
        
        # Read just the window (efficient)
        window = rasterio.windows.Window(
            col_start, row_start,
            col_end - col_start, row_end - row_start
        )
        buffer_data = raster_src.read(1, window=window)
        
        # Filter out nodata
        nodata = raster_src.nodata
        if nodata is not None:
            valid_data = buffer_data[buffer_data != nodata]
        else:
            valid_data = buffer_data[~np.isnan(buffer_data)]
        
        if len(valid_data) == 0:
            return None
        
        # Apply statistic
        if statistic == 'median':
            return float(np.median(valid_data))
        elif statistic == 'mean':
            return float(np.mean(valid_data))
        elif statistic == 'min':
            return float(np.min(valid_data))
        elif statistic == 'max':
            return float(np.max(valid_data))
        elif statistic == 'any':
            return int(np.any(valid_data > 0))
        else:
            return float(buffer_data[buffer_pixels, buffer_pixels])
    
    except Exception as e:
        return None


def extract_all_features(lat, lon, paths):
    """Extract all 5 features using smart buffered sampling."""
    features = {}
    
    # Each feature uses different statistic that makes sense for it
    feature_config = [
        ('elevation_m',           paths['dem'],       'median'),  # smooth out DEM noise
        ('upstream_catchment_km2', paths['flow'],     'max'),     # capture nearby drainage
        ('is_historical_paddy',   paths['paddy'],     'any'),     # any paddy nearby = paddy
        ('distance_to_river_m',   paths['river'],     'min'),     # closest river in buffer
        ('slope_degrees',         paths['slope'],     'median'),  # smooth slope noise
    ]
    
    for feat_name, path, stat in feature_config:
        try:
            with rasterio.open(path) as src:
                # Use larger buffer for paddy (higher resolution raster)
                buffer = 15 if feat_name == 'is_historical_paddy' else 5
                value = extract_with_buffer(src, lat, lon, buffer, stat)
                features[feat_name] = value if value is not None else 0
        except Exception as e:
            features[feat_name] = 0
    
    return features


def test_extraction():
    """Test the new buffered extraction on real Kerala places."""
    
    paths = {
        'dem': project_root / "data" / "raw" / "copernicus_dem" / "copernicus_glo30_kerala.tif",
        'flow': project_root / "data" / "processed" / "kerala_flow_accumulation_km2.tif",
        'paddy': project_root / "data" / "raw" / "lulc_2020" / "historical_paddy_kerala.tif",
        'river': project_root / "data" / "processed" / "distance_to_river_m.tif",
        'slope': project_root / "data" / "processed" / "slope_degrees.tif",
    }
    
    test_places = [
        ("Kuttanad",   9.4507, 76.4308),
        ("Aluva",     10.1078, 76.3569),
        ("Chalakudy", 10.3042, 76.3371),
        ("Munnar",    10.0870, 77.0601),
        ("Wayanad",   11.6094, 76.0838),
        ("Idukki",     9.8155, 76.9992),
        ("Pala",       9.7050, 76.6850),
        ("Edappally",  9.9893, 76.3060),
        ("Vagamon",    9.6783, 76.9067),
        ("Painavu",    9.8540, 76.9446),  # Better Idukki HQ
    ]
    
    print("\n" + "="*78)
    print("BUFFERED FEATURE EXTRACTION (5x5 pixel buffer = 150m radius)")
    print("="*78)
    print(f"\n{'Place':<15} {'Elev':<10} {'Catch':<8} {'River':<10} {'Slope':<8} {'Paddy':<7}")
    print("-"*78)
    
    import joblib
    import pandas as pd
    
    model = joblib.load(project_root / "models" / "flood_rf_model.pkl")
    
    results = []
    
    for name, lat, lon in test_places:
        feats = extract_all_features(lat, lon, paths)
        
        # Predict
        df_in = pd.DataFrame([[
            feats['elevation_m'],
            feats['upstream_catchment_km2'],
            feats['is_historical_paddy'],
            feats['distance_to_river_m'],
            feats['slope_degrees'],
        ]], columns=[
            'elevation_m', 'upstream_catchment_km2', 'is_historical_paddy',
            'distance_to_river_m', 'slope_degrees'
        ])
        
        prob = model.predict_proba(df_in)[0][1] * 100
        
        print(f"{name:<15} {feats['elevation_m']:<9.1f}m "
              f"{feats['upstream_catchment_km2']:<7.2f} "
              f"{feats['distance_to_river_m']:<9.0f}m "
              f"{feats['slope_degrees']:<7.1f}° "
              f"{feats['is_historical_paddy']:<6} → {prob:.1f}%")
        
        results.append((name, prob))
    
    print("\n" + "="*78)
    print("SANITY CHECK")
    print("="*78)
    
    # Compare against expected
    expected = {
        "Kuttanad": (80, 100),    # CRITICAL
        "Aluva": (50, 90),         # HIGH (improved from 1.7%)
        "Chalakudy": (40, 80),    # HIGH-MOD
        "Munnar": (0, 20),         # VERY LOW
        "Wayanad": (0, 25),        # LOW
        "Idukki": (0, 50),         # Town in hills
        "Pala": (10, 40),          # LOW-MOD
        "Edappally": (20, 60),    # LOW-MOD
        "Vagamon": (0, 25),        # LOW (hill area)
        "Painavu": (0, 25),        # LOW (hill HQ)
    }
    
    all_pass = True
    for name, prob in results:
        if name in expected:
            lo, hi = expected[name]
            status = "✅ PASS" if lo <= prob <= hi else "❌ FAIL"
            if status == "❌ FAIL":
                all_pass = False
            print(f"   {name:<15} got {prob:.1f}%, expected {lo}-{hi}% → {status}")
    
    if all_pass:
        print("\n🎉 ALL CHECKS PASS — Update streamlit_app.py to use buffered extraction")
    else:
        print("\n⚠️ Some still off — but should be much better than before")


if __name__ == "__main__":
    test_extraction()