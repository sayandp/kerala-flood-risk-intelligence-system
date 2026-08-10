# 🌧️ Kerala Flood Risk Intelligence

> AI-powered property flood risk assessment for Kerala using satellite imagery, climate science, and physics-based safety rules. Aligned with RBI's 2024 Climate Risk Disclosure Framework.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 The Problem

Kerala's 2018 floods destroyed **20,481 houses** and caused **₹40,000 crore** in damages. Yet today, when buying property in Kerala, there's **no consumer-facing tool** to check if a specific address sits in a high-risk flood zone.

Banks under RBI's 2024 climate disclosure framework now need this data for every loan. Real estate companies need it. Insurance needs it. Home buyers need it most.

## 🧠 What It Does

Type any Kerala property address → get an instant flood risk assessment combining:

- 🛰️ **5 satellite datasets** (Copernicus DEM, Sentinel-1 SAR, OSM waterways)
- 🤖 **Machine Learning model** (Random Forest, 91.96% accuracy)
- ⚖️ **12 physics-based safety rules** (Basel III pattern)
- 🌍 **IPCC climate projections** for 2030 and 2040
- 💰 **Financial impact** estimates (5-year)

## ✨ Features

- 🗺️ **3 Search Modes** — by address, GPS coordinates, or click-on-map
- 🔍 **Address Autosuggest** — real-time Kerala-specific suggestions  
- 🔥 **Kerala-wide Heatmap** — 3,479 pre-computed risk points across the state
- 📊 **SHAP Explainability** — see which features drove the prediction
- 🌐 **Real Satellite Imagery** — Esri World Imagery integration
- 📈 **Climate Timeline** — today vs 2030 vs 2040 projections
- 💎 **Modern UI** — glassmorphism, animated gradients, dark mode

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 91.96% |
| Recall | 91.92% |
| Precision | 91.92% |
| ROC-AUC | 0.975 |
| 5-fold Cross Validation | 94.83% ± 1.92% |

### Feature Importance
1. **Elevation** (50%) — height above sea level
2. **Slope** (42%) — terrain steepness
3. **River Distance** (5%) — nearest waterway
4. **Catchment** (3%) — upstream drainage area
5. **Paddy History** (0%) — reclaimed land indicator

## 🧪 Real-World Validation

| Location | Predicted | Reality |
|----------|-----------|---------|
| Kuttanad | 🚨 96.1% CRITICAL | Below sea level, chronic flooding |
| Aluva | 🟠 65.0% HIGH | 2018 flood epicenter |
| Edappally | 🚨 80.0% HIGH | Severely flooded 2018 |
| Pala | 🟡 45.0% MODERATE | Mid-altitude, near rivers |
| Munnar | ✅ 15.0% VERY LOW | High elevation (1457m) |
| Wayanad | ✅ 23.5% LOW | Hill station |
| Vagamon | ✅ 4.1% VERY LOW | Mountain plateau |

## 🏗️ Architecture

```
┌─────────────────────┐         ┌──────────────────────┐
│   React + Vite      │  HTTPS  │  FastAPI Backend     │
│   Tailwind CSS      │────────▶│  Python 3.10+        │
│   Framer Motion     │         │  Random Forest       │
│   Leaflet Maps      │◀────────│  Hybrid Rules        │
└─────────────────────┘         └──────────────────────┘
                                          │
                                          ▼
                            ┌──────────────────────────┐
                            │  Satellite Data          │
                            │  • Copernicus GLO-30 DEM │
                            │  • Sentinel-1 SAR (2018) │
                            │  • OSM Waterways (159k)  │
                            │  • Dynamic World LULC    │
                            └──────────────────────────┘
```

## 🛠️ Tech Stack

**Frontend:** React 18 + Vite + Tailwind CSS + Framer Motion + Leaflet  
**Backend:** FastAPI + scikit-learn + SHAP + Rasterio + Geopy  
**Data:** Copernicus GLO-30, Sentinel-1 SAR, OpenStreetMap, Dynamic World

## 🚀 Setup & Run Locally

### Prerequisites
- Python 3.10+
- Node.js 18+
- ~2 GB free disk space (for satellite data)

### 1. Clone the repo
```bash
git clone https://github.com/sayandp/kerala-flood-risk-intelligence-system.git
cd kerala-flood-risk-intelligence-system
```

### 2. Download data files

Satellite data (~1.5 GB) is too large for GitHub. Available on request:
- Email evaluator copy
- Or run `pipelines/01_download_rivers.py` through `08_recalculate_distance_to_water.py` to regenerate

Files needed in `data/`:
- `raw/copernicus_dem/copernicus_glo30_kerala.tif`
- `raw/lulc_2020/historical_paddy_kerala.tif`
- `raw/sentinel1_sar/sentinel1_flood_2018_kerala.tif`
- `raw/osm_kerala/kerala_waterways_osm.geojson`
- `processed/distance_to_river_m.tif`
- `processed/distance_to_water_m.tif`
- `processed/slope_degrees.tif`
- `processed/kerala_flow_accumulation_km2.tif`
- `processed/feature_matrix.csv`
- `processed/kerala_risk_grid.json`

### 3. Backend setup
```bash
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
cp frontend/.env.example frontend/.env
```
The defaults work as-is. Two settings matter if you hit a conflict:

| Variable | Default | Notes |
|---|---|---|
| `PORT` | `8001` | Backend port. Change it if 8001 is taken. |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Origins allowed to call the API. Must include the frontend URL. |
| `VITE_API_URL` | `http://localhost:8001` | Frontend → backend URL. Must match `PORT`. |

`.env` is gitignored — no credentials belong in the repo. Earth Engine
authentication (only needed to re-run the pipelines) uses your local
credential cache via `earthengine authenticate`.

### 5. Run backend
```bash
cd backend
python main.py
```
Backend runs on `http://localhost:8001`  
Auto-generated API docs: `http://localhost:8001/docs`

### 6. Frontend setup (new terminal)
```bash
cd frontend
npm install
npm run dev
```
App opens at `http://localhost:3000`

> **Note:** Vite reads `.env` only at startup. If you change `VITE_API_URL`,
> restart `npm run dev`.

## 📁 Project Structure

```
kerala-flood-risk/
├── README.md, requirements.txt, .gitignore
│
├── backend/
│   └── main.py                       FastAPI server with all endpoints
│
├── frontend/                         React app
│   ├── src/App.jsx                   Main component
│   ├── src/main.jsx                  Entry point
│   ├── src/index.css                 Tailwind + custom styles
│   └── package.json
│
├── src/                              Python library modules
│   ├── geocoder.py                   Address → GPS conversion
│   ├── climate_projection.py         IPCC scenarios
│   └── hybrid_risk_scorer.py         ML + rules logic
│
├── pipelines/                        Build scripts (run once)
│   ├── 01_download_rivers.py
│   ├── 02_calculate_distance_to_river.py
│   ├── 03_calculate_slope.py
│   ├── 04_build_training_dataset.py
│   ├── 05_retrain_model.py
│   ├── 06_smart_feature_extractor.py
│   ├── 07_download_osm_waterways.py
│   ├── 08_recalculate_distance_to_water.py
│   └── 09_compute_kerala_risk_grid.py
│
├── models/                           Trained models
│   ├── flood_rf_model.pkl
│   └── shap_explainer.pkl
│
├── data/                             Satellite data (not in git)
│   ├── raw/                          Source datasets
│   └── processed/                    ML-ready features
│
└── reports/                          Training plots & metrics
```

## 🧠 How It Works

### 1. Geocode Address
Convert "Kuttanad, Alappuzha" → GPS (9.4507°N, 76.4308°E) using OpenStreetMap Nominatim, validated within Kerala bounds.

### 2. Extract Satellite Features
Sample 5 features using a 150m smart buffer (handles GPS imprecision):

| Feature | Source | Resolution |
|---------|--------|-----------|
| Elevation | Copernicus GLO-30 DEM | 30m, ±2m vertical |
| Slope | Derived from DEM | 30m |
| Drainage Area | Flow accumulation | 30m |
| River Distance | 159,168 OSM waterways | <30m precision |
| Paddy History | Dynamic World LULC | 10m |

### 3. Hybrid Prediction
ML model gives raw probability → 12 physics rules check edge cases → higher value taken (safety-first approach).

**Example edge case:** Idukki Reservoir (725m elevation, on water)
- ML alone: 81.6% ❌ (false positive — model thought "on water = flooded")
- Rule: "high elevation + on water = reservoir, not flood zone"
- Final: 20% ✅ LOW

### 4. Climate Projection
Apply IPCC RCP 4.5 monsoon multipliers:
- **2030**: risk × 1.18
- **2040**: risk × 1.31

### 5. Financial Impact
5-year exposure based on average Kerala property values and historical flood damage data.

## 🎯 Use Cases

- 🏦 **Banks** — RBI 2024 climate risk disclosure for home loans
- 🏠 **Real Estate** — Property due diligence and pricing
- 🛡️ **Insurance** — Premium calculation and underwriting
- 🏘️ **Home Buyers** — Make informed property decisions
- 🏛️ **Government** — Urban planning, disaster preparedness

## 🔬 Hybrid ML + Rules Architecture

Standard pattern in regulated finance (Basel III credit risk frameworks):

```python
# Pure ML can fail on edge cases
ml_prob = model.predict_proba(features)[0][1]

# Rules catch what ML misses
rule_result = apply_business_rules(features, ml_prob)

# Final = max of both (safety-first)
final = rule_result['final_probability']
```

Why this matters: a regulator (or bank) can audit each rule. Pure ML is a black box. Hybrid is **explainable and defensible**.

## 📚 Data Citations

- **Copernicus GLO-30 DEM** — © European Space Agency (ESA), 2024
- **Sentinel-1 SAR** — Copernicus Sentinel data
- **OpenStreetMap** — © OpenStreetMap contributors
- **Dynamic World v1** — Brown et al., Google & WRI (2022)
- **IPCC AR6** — Working Group I Report (2021)

## 👤 Author

**Sayand P**  
Post Graduate Diploma in Data Science  
University of Calicut

📧 sayandp0@gmail.com  
💼 [LinkedIn](https://www.linkedin.com/in/sayand-p-2215a9225/)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details

---

<div align="center">

**Built for Kerala. By a student. Engineered like a bank.**

⭐ Star this repo if it helped you!

</div>