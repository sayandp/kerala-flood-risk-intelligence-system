"""
============================================================
backend/main.py
FastAPI backend for Kerala Flood Risk Scorer
============================================================
Endpoints:
  GET  /                       - Health check
  POST /api/predict            - Predict flood risk (main endpoint)
  POST /api/geocode            - Geocode an address
  GET  /api/health             - Detailed health check
  GET  /api/suggest            - Address autosuggest
  GET  /api/risk-grid          - Kerala-wide heatmap grid
  POST /api/predict-by-address - Geocode + predict in one call

Run locally:
  uvicorn main:app --reload --port 8000

Then visit: http://localhost:8000/docs (auto-generated API docs)
============================================================
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path so we can import from src/
backend_dir  = Path(__file__).parent
project_root = backend_dir.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from dotenv import load_dotenv

# Load .env from the project root (gitignored — see .env.example for the keys)
load_dotenv(project_root / ".env")

logger = logging.getLogger("kerala_flood_risk")

# Import our existing modules
from src.geocoder import KeralaGeocoder
from src.climate_projection import (
    project_future_risk,
    calculate_financial_impact,
    generate_recommendation,
    get_risk_band,
)
from src.hybrid_risk_scorer import apply_business_rules


# ─── Configuration ────────────────────────────────────────
MODEL_PATH      = project_root / "models"     / "flood_rf_model.pkl"
SHAP_PATH       = project_root / "models"     / "shap_explainer.pkl"
DEM_PATH        = project_root / "data" / "raw" / "copernicus_dem"   / "copernicus_glo30_kerala.tif"
FLOW_PATH       = project_root / "data" / "processed"                / "kerala_flow_accumulation_km2.tif"
PADDY_PATH      = project_root / "data" / "raw" / "lulc_2020"        / "historical_paddy_kerala.tif"
RIVER_DIST_PATH = project_root / "data" / "processed"                / "distance_to_river_m.tif"
SLOPE_PATH      = project_root / "data" / "processed"                / "slope_degrees.tif"

FEATURE_COLS = [
    'elevation_m',
    'upstream_catchment_km2',
    'is_historical_paddy',
    'distance_to_river_m',
    'slope_degrees',
]

# Elevation filter for paddy detection — Kerala paddy is by definition low-lying.
# Anything classified as "crops" above this elevation is almost always a
# misclassified tea plantation or terraced field in the Western Ghats.
PADDY_MAX_ELEVATION_M = 50.0


# ─── FastAPI App ──────────────────────────────────────────
app = FastAPI(
    title="Kerala Flood Risk Scorer API",
    description="Hybrid ML + Rules property flood risk assessment for Kerala",
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────
# Explicit origins only. allow_origins=["*"] together with
# allow_credentials=True makes Starlette reflect any caller's Origin back
# with Access-Control-Allow-Credentials: true, which lets any website make
# credentialed calls to this API. The frontend sends no cookies or auth
# headers, so credentials stay off.
#
# Override for deployment by setting ALLOWED_ORIGINS in .env as a
# comma-separated list, e.g. ALLOWED_ORIGINS=https://your-app.vercel.app
DEFAULT_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ─── Load Models on Startup ───────────────────────────────
print("Loading models...")
try:
    model     = joblib.load(MODEL_PATH)
    explainer = joblib.load(SHAP_PATH)
    geocoder  = KeralaGeocoder()
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    raise


# ─── Request/Response Models ──────────────────────────────
class GeocodeRequest(BaseModel):
    address: str = Field(..., min_length=2, example="Kuttanad, Alappuzha")

class PredictRequest(BaseModel):
    latitude:  float = Field(..., ge=8.0,  le=12.8, example=9.4507)
    longitude: float = Field(..., ge=74.5, le=77.5, example=76.4308)

class PredictByAddressRequest(BaseModel):
    address: str = Field(..., min_length=2, example="Kuttanad, Alappuzha")

class GeocodeResponse(BaseModel):
    success:      bool
    latitude:     Optional[float] = None
    longitude:    Optional[float] = None
    full_address: Optional[str]   = None
    error:        Optional[str]   = None
    is_in_kerala: bool = False


# ─── Helper: Smart Buffer Extraction ─────────────────────
def extract_with_buffer(src, lat, lon, buffer=5, stat='median'):
    """
    Extract a raster feature using a buffered window around (lat, lon).

    Args:
        src    : open rasterio dataset
        lat    : latitude  (degrees)
        lon    : longitude (degrees)
        buffer : half-window size in pixels
        stat   : aggregation — 'median', 'min', 'max', 'any'

    Returns:
        float or int, or 0 on failure
    """
    try:
        r, c = src.index(lon, lat)   # NOTE: rasterio takes (lon, lat)

        col_off = max(0, c - buffer)
        row_off = max(0, r - buffer)
        col_end = min(src.width,  c + buffer + 1)
        row_end = min(src.height, r + buffer + 1)
        win_w   = col_end - col_off
        win_h   = row_end - row_off

        if win_w <= 0 or win_h <= 0:
            return 0

        window = Window(col_off, row_off, win_w, win_h)
        data   = src.read(1, window=window)

        # ── Mask nodata / NaN values ─────────────────────
        nodata = src.nodata
        if nodata is not None:
            valid = data[data != nodata]
        elif np.issubdtype(data.dtype, np.floating):
            valid = data[~np.isnan(data)]
        else:
            # Integer raster with no nodata (e.g. paddy) — use all pixels
            valid = data.flatten()

        if len(valid) == 0:
            return 0

        if stat == 'median': return float(np.median(valid))
        if stat == 'max':    return float(np.max(valid))
        if stat == 'min':    return float(np.min(valid))
        if stat == 'any':    return int(np.any(valid > 0))
        return 0

    except Exception as e:
        print(f"[extract_with_buffer] ERROR at ({lat},{lon}) stat={stat}: {e}")
        return 0


# ─── Helper: Smart Paddy Extraction with Elevation Filter ──
def extract_paddy_smart(paddy_src, dem_src, lat, lon, buf=50, max_elev=PADDY_MAX_ELEVATION_M):
    """
    Detect paddy land within a 500m radius, BUT only if local elevation is low.

    Why the elevation filter:
      Dynamic World's 'crops' class sometimes mislabels Western Ghats tea
      plantations and terraced fields as paddy. Kerala paddy land is by
      definition low-lying (Kuttanad, Alappuzha coastal belt, river plains).
      So we only count paddy pixels where local elevation is below 50m —
      this filters out high-altitude false positives without missing any
      real paddy land.

    Args:
        paddy_src : open rasterio dataset for the paddy raster (10m res)
        dem_src   : open rasterio dataset for the elevation raster (30m res)
        lat, lon  : query coordinates
        buf       : half-window size in pixels for paddy search (default 50 → 500m)
        max_elev  : elevation ceiling above which paddy detection is suppressed

    Returns:
        1 if low-elevation paddy detected nearby, else 0
    """
    try:
        # First check elevation at this location (cheap — single pixel read)
        r_d, c_d = dem_src.index(lon, lat)
        elev_data = dem_src.read(1, window=Window(c_d, r_d, 1, 1))

        if elev_data.size == 0:
            return 0

        elev = float(elev_data[0, 0])

        # If we're in the highlands, no paddy regardless of what the raster says
        if elev > max_elev:
            return 0

        # Otherwise look for paddy pixels within the 500m buffer
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

    except Exception as e:
        print(f"[extract_paddy_smart] ERROR at ({lat},{lon}): {e}")
        return 0


# ─── Feature Extraction ───────────────────────────────────
def extract_features(lat, lon):
    """
    Extract all 5 model features at a given location.

    Buffer sizes are tuned to each raster's resolution:
      DEM / slope / distance  → 30m res  → buffer=5  → 150m radius
      Flow accumulation       → 30m res  → buffer=5  → 150m radius
      Paddy (Dynamic World)   → 10m res  → buffer=50 → 500m radius
        + elevation filter to suppress Western Ghats false positives.
    """
    features = {}

    with rasterio.open(DEM_PATH) as src:
        features['elevation_m'] = extract_with_buffer(
            src, lat, lon, buffer=5, stat='median')

    with rasterio.open(FLOW_PATH) as src:
        features['upstream_catchment_km2'] = extract_with_buffer(
            src, lat, lon, buffer=5, stat='max')

    # ── SMART PADDY DETECTION ──────────────────────────────
    # Open paddy AND DEM together so we can apply an elevation filter.
    # This catches Kuttanad/Chalakudy paddy land correctly while
    # rejecting Munnar/Wayanad tea-plantation false positives.
    with rasterio.open(PADDY_PATH) as paddy_src, rasterio.open(DEM_PATH) as dem_src:
        features['is_historical_paddy'] = extract_paddy_smart(
            paddy_src, dem_src, lat, lon,
            buf=50, max_elev=PADDY_MAX_ELEVATION_M)

    with rasterio.open(RIVER_DIST_PATH) as src:
        features['distance_to_river_m'] = extract_with_buffer(
            src, lat, lon, buffer=5, stat='min')

    with rasterio.open(SLOPE_PATH) as src:
        features['slope_degrees'] = extract_with_buffer(
            src, lat, lon, buffer=5, stat='median')

    return features


# ─── API Endpoints ────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "Kerala Flood Risk Scorer API",
        "version": "1.0.0",
        "status":  "online",
        "docs":    "/docs",
    }


@app.get("/api/health")
def health():
    files_ok = all([
        MODEL_PATH.exists(),
        SHAP_PATH.exists(),
        DEM_PATH.exists(),
        FLOW_PATH.exists(),
        PADDY_PATH.exists(),
        RIVER_DIST_PATH.exists(),
        SLOPE_PATH.exists(),
    ])
    return {
        "status":            "healthy" if files_ok else "degraded",
        "model_loaded":      model     is not None,
        "explainer_loaded":  explainer is not None,
        "all_files_present": files_ok,
        "feature_count":     len(FEATURE_COLS),
        "feature_names":     FEATURE_COLS,
    }


@app.post("/api/geocode", response_model=GeocodeResponse)
def geocode_address(request: GeocodeRequest):
    result = geocoder.geocode(request.address)
    return GeocodeResponse(**result)


@app.get("/api/suggest")
def suggest_addresses(q: str):
    """Real-time address autosuggest filtered to Kerala."""
    if len(q) < 2:
        return {"suggestions": []}
    try:
        import requests as req
        response = req.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                'q':              f'{q}, Kerala, India',
                'format':         'json',
                'limit':          8,
                'countrycodes':   'in',
                'addressdetails': 1,
                'bounded':        1,
                'viewbox':        '74.5,12.8,77.5,8.0',
            },
            headers={'User-Agent': 'kerala_flood_risk_app/1.0'},
            timeout=5,
        )
        if response.status_code != 200:
            return {"suggestions": []}

        suggestions = []
        for r in response.json():
            address = r.get('address', {})
            if 'kerala' not in address.get('state', '').lower():
                continue
            name  = r.get('display_name', '')
            parts = name.split(',')[:3]
            short = ', '.join(parts).strip()
            suggestions.append({
                'display_name': short,
                'full_name':    name,
                'latitude':     float(r.get('lat', 0)),
                'longitude':    float(r.get('lon', 0)),
                'type':         r.get('type', ''),
                'category':     r.get('category', ''),
            })

        return {"suggestions": suggestions[:6]}
    except Exception:
        logger.exception("Address suggestion failed for query %r", q)
        return {"suggestions": []}


@app.get("/api/risk-grid")
def get_risk_grid():
    """Returns precomputed Kerala-wide flood risk grid for heatmap."""
    import json
    grid_path = project_root / "data" / "processed" / "kerala_risk_grid.json"
    if not grid_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Risk grid not generated. Run pipelines/09_compute_kerala_risk_grid.py first",
        )
    try:
        with open(grid_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception:
        # Log the real error server-side; rasterio/json errors embed absolute
        # filesystem paths that should not reach the client.
        logger.exception("Failed to read risk grid at %s", grid_path)
        raise HTTPException(status_code=500, detail="Failed to load risk grid")


@app.post("/api/predict")
def predict_flood_risk(request: PredictRequest):
    """
    Main prediction endpoint.
    Takes lat/lon coordinates, returns full flood risk assessment.
    """
    lat = request.latitude
    lon = request.longitude

    try:
        # 1. Extract satellite features
        features = extract_features(lat, lon)

        # 2. Build feature dataframe
        feature_array = np.array([[
            features['elevation_m'],
            features['upstream_catchment_km2'],
            features['is_historical_paddy'],
            features['distance_to_river_m'],
            features['slope_degrees'],
        ]])
        feature_df = pd.DataFrame(feature_array, columns=FEATURE_COLS)

        # 3. ML model prediction
        ml_probability = float(model.predict_proba(feature_df)[0][1])

        # 4. Apply hybrid physics rules
        rule_result       = apply_business_rules(features, ml_probability)
        final_probability = rule_result['final_probability']

        # 5. Classify risk band
        risk_pct = final_probability * 100
        severity = get_risk_band(risk_pct)

        # 6. Climate projections (IPCC RCP 4.5)
        projections = project_future_risk(final_probability)

        # 7. Financial impact estimates
        finances = calculate_financial_impact(final_probability)

        # 8. Recommendation text
        recommendation = generate_recommendation(
            projections['today']['probability'],
            projections['2040']['probability'],
        )

        # 9. SHAP explainability
        shap_explanation = None
        try:
            shap_values = explainer(feature_df)
            if hasattr(shap_values, 'values'):
                vals = (
                    shap_values.values[0]
                    if shap_values.values.ndim == 2
                    else shap_values.values[0, :, 1]
                )
                shap_explanation = [
                    {
                        'feature':      name,
                        'value':        float(features[name]),
                        'shap_value':   float(v),
                        'contribution': 'increases risk' if v > 0 else 'decreases risk',
                    }
                    for name, v in zip(FEATURE_COLS, vals)
                ]
        except Exception:
            pass  # SHAP is optional — prediction still succeeds without it

        # 10. Build and return response
        return {
            'success': True,
            'coordinates': {
                'latitude':  lat,
                'longitude': lon,
            },
            'risk': {
                'final_probability':  final_probability,
                'final_percentage':   round(risk_pct, 1),
                'ml_probability':     ml_probability,
                'ml_percentage':      round(ml_probability * 100, 1),
                'severity':           severity['label'],
                'color':              severity['color'],
                'emoji':              severity['emoji'],
                'used_rule_override': rule_result['used_rule_override'],
                'rule_adjustments': [
                    {
                        'type':     adj['type'],
                        'reason':   adj['reason'],
                        'from_pct': round(adj['from'] * 100, 1),
                        'to_pct':   round(adj['to']   * 100, 1),
                    }
                    for adj in rule_result['rule_adjustments']
                ],
            },
            'features': {
                'elevation_m':            round(features['elevation_m'],            1),
                'upstream_catchment_km2': round(features['upstream_catchment_km2'], 3),
                'is_historical_paddy':    bool(features['is_historical_paddy']),
                'distance_to_river_m':    round(features['distance_to_river_m'],    0),
                'slope_degrees':          round(features['slope_degrees'],           1),
            },
            'climate_projections': {
                year: {
                    'probability': data['probability'],
                    'severity':    data['severity'],
                    'color':       data['color'],
                    'emoji':       data['emoji'],
                    'description': data['description'],
                }
                for year, data in projections.items()
            },
            'financial_impact': {
                'expected_rebuilding_loss':  finances['expected_rebuilding_loss'],
                'insurance_extra_yearly':    finances['insurance_extra_yearly'],
                'resale_value_loss':         finances['resale_value_loss'],
                'total_5yr_impact':          finances['total_5yr_impact'],
                'rebuilding_loss_str':       finances['rebuilding_loss_str'],
                'insurance_str':             finances['insurance_str'],
                'resale_loss_str':           finances['resale_loss_str'],
                'total_5yr_str':             finances['total_5yr_str'],
            },
            'recommendation':   recommendation,
            'shap_explanation': shap_explanation,
        }

    except Exception:
        logger.exception("Prediction failed at (%s, %s)", lat, lon)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/api/predict-by-address")
def predict_by_address(request: PredictByAddressRequest):
    """Geocode an address then predict flood risk — convenience endpoint."""
    geo_result = geocoder.geocode(request.address)

    if not geo_result['success']:
        raise HTTPException(status_code=404, detail=geo_result['error'])
    if not geo_result['is_in_kerala']:
        raise HTTPException(status_code=400, detail="Address is outside Kerala")

    pred_request = PredictRequest(
        latitude=geo_result['latitude'],
        longitude=geo_result['longitude'],
    )
    prediction = predict_flood_risk(pred_request)
    prediction['geocoding'] = {
        'queried_address': request.address,
        'matched_address': geo_result['full_address'],
    }
    return prediction


if __name__ == "__main__":
    import uvicorn

    # Bind to loopback by default. Set HOST=0.0.0.0 in .env to expose on the LAN.
    #
    # PORT defaults to 8001 because another local project occupies 8000. Windows
    # lets a 127.0.0.1 listener and a 0.0.0.0 listener share a port, and the more
    # specific one wins — so a collision silently routes browser traffic to the
    # wrong app instead of raising "address already in use".
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
    )