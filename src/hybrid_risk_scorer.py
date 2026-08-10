"""
============================================================
src/hybrid_risk_scorer.py — VERSION 2
============================================================
Hybrid Flood Risk Scorer = ML Model + Physics Rules

Updates from v1:
  - Added rule for "high elevation + on water body" (Idukki dam case)
  - Adjusted ceiling threshold for hill towns
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ─── PHYSICS-BASED RULES ──────────────────────────────────
RISK_RULES = [
    # ─── EXTREME RISK (>=80%) ─────────────────────────────
    {
        'condition': lambda f: f['elevation_m'] < 5,
        'risk_floor': 0.85,
        'reason': "Below 5m elevation — extreme flood risk in Kerala monsoons"
    },
    {
        'condition': lambda f: f['elevation_m'] < 10 and f['distance_to_river_m'] < 500,
        'risk_floor': 0.80,
        'reason': "Low elevation (<10m) AND close to river (<500m)"
    },
    {
        'condition': lambda f: f['elevation_m'] < 20 and f['is_historical_paddy'] == 1,
        'risk_floor': 0.78,
        'reason': "Built on reclaimed paddy field below 20m"
    },
    
    # ─── HIGH RISK (60-80%) ───────────────────────────────
    {
        'condition': lambda f: f['elevation_m'] < 20 and f['distance_to_river_m'] < 1000 and f['slope_degrees'] < 5,
        'risk_floor': 0.65,
        'reason': "Low elevation, near river, flat terrain — classic flood zone"
    },
    {
        'condition': lambda f: f['elevation_m'] < 30 and f['distance_to_river_m'] < 500,
        'risk_floor': 0.60,
        'reason': "Low elevation AND very close to river"
    },
    
    # ─── MODERATE RISK (40-60%) ───────────────────────────
    {
        'condition': lambda f: f['elevation_m'] < 50 and f['distance_to_river_m'] < 1000 and f['slope_degrees'] < 8,
        'risk_floor': 0.45,
        'reason': "Low-moderate elevation in floodplain"
    },
    
    # ─── HARD CAPS (force LOW risk) ───────────────────────
    {
        'condition': lambda f: f['elevation_m'] > 800,
        'risk_ceiling': 0.15,
        'reason': "High elevation (>800m) — physical impossibility of monsoon flooding"
    },
    {
        'condition': lambda f: f['elevation_m'] > 500 and f['slope_degrees'] > 10,
        'risk_ceiling': 0.20,
        'reason': "Mountain slope — water flows away rapidly"
    },
    {
        'condition': lambda f: f['elevation_m'] > 300 and f['distance_to_river_m'] > 2000,
        'risk_ceiling': 0.25,
        'reason': "Elevated AND far from rivers"
    },
    
    # ─── NEW: HILL ZONE RULES (Idukki, Wayanad fix) ───────
    {
        'condition': lambda f: f['elevation_m'] > 600 and f['slope_degrees'] < 1 and f['distance_to_river_m'] < 200,
        'risk_ceiling': 0.20,
        'reason': "High elevation reservoir/dam area — not flood-prone (water body itself)"
    },
    {
        'condition': lambda f: 500 < f['elevation_m'] < 1000 and f['slope_degrees'] > 5,
        'risk_ceiling': 0.30,
        'reason': "Hill zone with sloped terrain — drains naturally"
    },
    {
        'condition': lambda f: f['elevation_m'] > 700,
        'risk_ceiling': 0.25,
        'reason': "Hill town zone (>700m elevation) — minimal flood risk"
    },
]


def apply_business_rules(features: dict, ml_probability: float) -> dict:
    """Apply physics-based rules to ML model output."""
    final_prob = ml_probability
    rules_applied = []
    rule_adjustments = []
    
    # First pass: floors (force minimum risk)
    for rule in RISK_RULES:
        if 'risk_floor' not in rule:
            continue
        try:
            if rule['condition'](features):
                rules_applied.append(rule['reason'])
                if final_prob < rule['risk_floor']:
                    old_prob = final_prob
                    final_prob = rule['risk_floor']
                    rule_adjustments.append({
                        'type': 'floor',
                        'reason': rule['reason'],
                        'from': old_prob,
                        'to': final_prob,
                    })
        except (KeyError, TypeError):
            continue
    
    # Second pass: ceilings (force maximum risk)
    for rule in RISK_RULES:
        if 'risk_ceiling' not in rule:
            continue
        try:
            if rule['condition'](features):
                rules_applied.append(rule['reason'])
                if final_prob > rule['risk_ceiling']:
                    old_prob = final_prob
                    final_prob = rule['risk_ceiling']
                    rule_adjustments.append({
                        'type': 'ceiling',
                        'reason': rule['reason'],
                        'from': old_prob,
                        'to': final_prob,
                    })
        except (KeyError, TypeError):
            continue
    
    if rule_adjustments:
        if len(rule_adjustments) == 1:
            adj = rule_adjustments[0]
            if adj['type'] == 'floor':
                explanation = f"Model said {adj['from']*100:.1f}% but rule applied: {adj['reason']}. Adjusted to {adj['to']*100:.1f}%."
            else:
                explanation = f"Model said {adj['from']*100:.1f}% but rule capped: {adj['reason']}. Adjusted to {adj['to']*100:.1f}%."
        else:
            explanation = f"Model said {ml_probability*100:.1f}%. Multiple rules applied. Final: {final_prob*100:.1f}%."
    else:
        explanation = f"ML model prediction used as-is: {ml_probability*100:.1f}% (no rule adjustments)"
    
    return {
        'final_probability': final_prob,
        'ml_probability': ml_probability,
        'rules_applied': rules_applied,
        'rule_adjustments': rule_adjustments,
        'explanation': explanation,
        'used_rule_override': len(rule_adjustments) > 0,
    }


# ─── Test ────────────────────────────────────────────────
if __name__ == "__main__":
    import joblib
    import pandas as pd
    import rasterio
    import numpy as np
    
    project_root = Path(__file__).parent.parent
    model = joblib.load(project_root / "models" / "flood_rf_model.pkl")
    
    paths = {
        'dem': project_root / "data" / "raw" / "copernicus_dem" / "copernicus_glo30_kerala.tif",
        'flow': project_root / "data" / "processed" / "kerala_flow_accumulation_km2.tif",
        'paddy': project_root / "data" / "raw" / "lulc_2020" / "historical_paddy_kerala.tif",
        'river': project_root / "data" / "processed" / "distance_to_river_m.tif",
        'slope': project_root / "data" / "processed" / "slope_degrees.tif",
    }
    
    def extract_with_buffer(src, lat, lon, buffer=5, stat='median'):
        try:
            r, c = src.index(lon, lat)
            window = rasterio.windows.Window(
                max(0, c - buffer), max(0, r - buffer),
                min(src.width, c + buffer + 1) - max(0, c - buffer),
                min(src.height, r + buffer + 1) - max(0, r - buffer)
            )
            data = src.read(1, window=window)
            nodata = src.nodata
            valid = data[data != nodata] if nodata is not None else data[~np.isnan(data)]
            if len(valid) == 0: return 0
            if stat == 'median': return float(np.median(valid))
            elif stat == 'max': return float(np.max(valid))
            elif stat == 'min': return float(np.min(valid))
            elif stat == 'any': return int(np.any(valid > 0))
        except: return 0
    
    def get_features(lat, lon):
        features = {}
        with rasterio.open(paths['dem']) as src:
            features['elevation_m'] = extract_with_buffer(src, lat, lon, 5, 'median')
        with rasterio.open(paths['flow']) as src:
            features['upstream_catchment_km2'] = extract_with_buffer(src, lat, lon, 5, 'max')
        with rasterio.open(paths['paddy']) as src:
            features['is_historical_paddy'] = extract_with_buffer(src, lat, lon, 15, 'any')
        with rasterio.open(paths['river']) as src:
            features['distance_to_river_m'] = extract_with_buffer(src, lat, lon, 5, 'min')
        with rasterio.open(paths['slope']) as src:
            features['slope_degrees'] = extract_with_buffer(src, lat, lon, 5, 'median')
        return features
    
    test_places = [
        ("Kuttanad",   9.4507, 76.4308, "CRITICAL"),
        ("Aluva",     10.1078, 76.3569, "HIGH"),
        ("Chalakudy", 10.3042, 76.3371, "HIGH/MODERATE"),
        ("Munnar",    10.0870, 77.0601, "VERY LOW"),
        ("Wayanad",   11.6094, 76.0838, "LOW"),
        ("Idukki",     9.8155, 76.9992, "LOW"),
        ("Pala",       9.7050, 76.6850, "MODERATE"),
        ("Edappally",  9.9893, 76.3060, "MODERATE"),
        ("Vagamon",    9.6783, 76.9067, "VERY LOW"),
        ("Painavu",    9.8540, 76.9446, "VERY LOW"),
    ]
    
    print("\n" + "="*80)
    print("HYBRID RISK SCORER V2 (ML + Improved Rules)")
    print("="*80)
    print(f"\n{'Place':<14} {'Elev':<8} {'River':<8} {'Slope':<7} {'ML':<8} {'Final':<8} {'Match':<10}")
    print("-"*80)
    
    for name, lat, lon, expected in test_places:
        feats = get_features(lat, lon)
        df_in = pd.DataFrame([[
            feats['elevation_m'], feats['upstream_catchment_km2'],
            feats['is_historical_paddy'], feats['distance_to_river_m'],
            feats['slope_degrees'],
        ]], columns=['elevation_m', 'upstream_catchment_km2', 'is_historical_paddy', 'distance_to_river_m', 'slope_degrees'])
        
        ml_prob = model.predict_proba(df_in)[0][1]
        result = apply_business_rules(feats, ml_prob)
        
        # Check if final matches expected
        final_pct = result['final_probability'] * 100
        if expected == "CRITICAL" and final_pct >= 80: match = "✅"
        elif expected == "HIGH" and 60 <= final_pct <= 90: match = "✅"
        elif expected == "HIGH/MODERATE" and 50 <= final_pct <= 80: match = "✅"
        elif expected == "MODERATE" and 30 <= final_pct <= 70: match = "✅"
        elif expected == "LOW" and final_pct <= 35: match = "✅"
        elif expected == "VERY LOW" and final_pct <= 25: match = "✅"
        else: match = "❌"
        
        print(f"{name:<14} {feats['elevation_m']:>5.0f}m  "
              f"{feats['distance_to_river_m']:>5.0f}m  "
              f"{feats['slope_degrees']:>4.1f}°   "
              f"{ml_prob*100:>5.1f}%   "
              f"{final_pct:>5.1f}%   "
              f"{match} {expected}")
        
        if result['used_rule_override']:
            for adj in result['rule_adjustments']:
                arrow = "⬆" if adj['type'] == 'floor' else "⬇"
                print(f"   {arrow} {adj['type'].upper()}: {adj['reason'][:65]}")
    
    print("\n" + "="*80)