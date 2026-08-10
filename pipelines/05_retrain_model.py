"""
============================================================
src/05_retrain_model.py
============================================================
STEP 5 of dataset rebuild — TRAIN THE FINAL MODEL

Trains a Random Forest using the new 5-feature dataset:
  1. elevation_m
  2. upstream_catchment_km2
  3. is_historical_paddy
  4. distance_to_river_m  (NEW!)
  5. slope_degrees        (NEW!)

OUTPUT:
  - models/flood_rf_model.pkl     (replaces old broken one)
  - models/shap_explainer.pkl     (matched explainer)
  - reports/training_report.txt   (performance metrics)
============================================================
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
)
import shap


# ─── Paths ────────────────────────────────────────────────
DATA_PATH = project_root / "data" / "processed" / "feature_matrix_v2.csv"
MODEL_PATH = project_root / "models" / "flood_rf_model.pkl"
SHAP_PATH = project_root / "models" / "shap_explainer.pkl"
REPORT_PATH = project_root / "reports" / "training_report.txt"

# Feature columns (in order — model expects this exact order)
FEATURE_COLS = [
    'elevation_m',
    'upstream_catchment_km2',
    'is_historical_paddy',
    'distance_to_river_m',
    'slope_degrees',
]


def load_data():
    """Load and prepare training data."""
    print("\n📂 Loading training data...")
    df = pd.read_csv(DATA_PATH)
    
    print(f"   Total rows: {len(df)}")
    print(f"   Class distribution:")
    print(f"     Flooded:    {(df['flooded']==1).sum()}")
    print(f"     Safe:       {(df['flooded']==0).sum()}")
    
    X = df[FEATURE_COLS]
    y = df['flooded']
    
    return X, y


def train_model(X, y):
    """Train Random Forest with proper hyperparameters."""
    print("\n🎯 Splitting train/test (80/20 stratified)...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    print("\n🤖 Training Random Forest...")
    print("   Hyperparameters:")
    print("     n_estimators=300 (lots of trees for stability)")
    print("     max_depth=12 (prevents overfitting)")
    print("     min_samples_split=10 (requires solid evidence)")
    print("     class_weight='balanced' (forces equal learning)")
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Trained")
    
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, X_train, y_train):
    """Comprehensive evaluation."""
    print("\n📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Basic metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\n   ━━━ TEST SET PERFORMANCE ━━━")
    print(f"   Accuracy:   {acc*100:.2f}%")
    print(f"   Recall:     {recall*100:.2f}%  (catches actual floods)")
    print(f"   Precision:  {precision*100:.2f}%  (avoids false alarms)")
    print(f"   F1 Score:   {f1:.3f}")
    print(f"   ROC-AUC:    {auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   Confusion Matrix:")
    print(f"                Predicted Safe  Predicted Flood")
    print(f"   Actual Safe       {cm[0][0]:>5}              {cm[0][1]:>5}")
    print(f"   Actual Flood      {cm[1][0]:>5}              {cm[1][1]:>5}")
    
    # Cross-validation
    print(f"\n   ━━━ 5-FOLD CROSS VALIDATION ━━━")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"   Fold accuracies: {[f'{s*100:.1f}%' for s in cv_scores]}")
    print(f"   Mean CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    
    # Feature importance
    print(f"\n   ━━━ FEATURE IMPORTANCE ━━━")
    importances = list(zip(FEATURE_COLS, model.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)
    
    for name, imp in importances:
        bar = "█" * int(imp * 40)
        print(f"   {name:<28} {imp:.4f}  {bar}")
    
    return {
        'accuracy': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'importances': importances,
    }


def synthetic_sanity_check(model):
    """Test obvious cases — model MUST get these right."""
    print("\n🧪 SANITY CHECK — Testing physical scenarios...")
    
    test_cases = [
        # (elevation, catchment, paddy, river_dist, slope, description, expected)
        ([1500, 0.01, 0, 5000, 25], "Munnar (mountain)",      "<10%"),
        ([1000, 0.05, 0, 2000, 15], "Vagamon (hills)",        "<20%"),
        ([500, 0.5, 0, 1500, 8], "Mid-elevation hill",        "20-50%"),
        ([100, 1.5, 0, 300, 2], "Foothill near river",        "40-70%"),
        ([20, 5.0, 1, 100, 1], "Chalakudy floodplain",        "70%+"),
        ([2, 20.0, 1, 50, 0.5], "Kuttanad",                   "85%+"),
    ]
    
    print(f"\n   {'Scenario':<32} {'Risk':<10} {'Expected':<12} {'Status':<10}")
    print("   " + "-" * 70)
    
    all_pass = True
    for features, desc, expected in test_cases:
        df_test = pd.DataFrame([features], columns=FEATURE_COLS)
        prob = model.predict_proba(df_test)[0][1] * 100
        
        # Check if reasonable
        elev = features[0]
        if elev > 800 and prob > 30:
            status = "❌ FAIL"
            all_pass = False
        elif elev < 50 and prob < 50:
            status = "❌ FAIL"
            all_pass = False
        elif 50 <= elev <= 800 and (prob < 10 or prob > 95):
            status = "⚠️ ODD"
        else:
            status = "✅ PASS"
        
        print(f"   {desc:<32} {prob:>5.1f}%   {expected:<12} {status}")
    
    if all_pass:
        print("\n   🎉 ALL PHYSICAL CHECKS PASSED")
    else:
        print("\n   ⚠️ Some checks failed — model may need more features")
    
    return all_pass


def real_world_test(model):
    """Test with actual Kerala addresses (will need raster lookup later)."""
    print("\n🌏 Real Kerala test cases (using known feature values):")
    
    # These are realistic feature combinations for known places
    real_tests = [
        ("Kuttanad",       [1, 15.0, 1, 50, 0.5]),
        ("Aluva",          [10, 8.0, 0, 100, 1.0]),
        ("Chalakudy",      [25, 5.0, 0, 200, 1.5]),
        ("Edappally",      [10, 0.5, 0, 250, 4.0]),
        ("Pala",           [80, 1.0, 0, 400, 5.0]),
        ("Wayanad town",   [780, 0.1, 0, 1500, 10.0]),
        ("Munnar",         [1500, 0.05, 0, 100, 5.0]),
    ]
    
    print(f"\n   {'Location':<22} {'Risk':<10}")
    print("   " + "-" * 35)
    for name, features in real_tests:
        df_test = pd.DataFrame([features], columns=FEATURE_COLS)
        prob = model.predict_proba(df_test)[0][1] * 100
        
        if prob >= 80:
            tag = "🚨 CRITICAL"
        elif prob >= 60:
            tag = "🟠 HIGH"
        elif prob >= 40:
            tag = "🟡 MODERATE"
        elif prob >= 20:
            tag = "🟢 LOW"
        else:
            tag = "✅ VERY LOW"
        
        print(f"   {name:<22} {prob:>5.1f}%   {tag}")


def save_model_and_explainer(model, X_train, metrics):
    """Save model, SHAP explainer, and training report."""
    print("\n💾 Saving model and explainer...")
    
    # Backup old files
    if MODEL_PATH.exists():
        backup = MODEL_PATH.with_suffix('.pkl.OLD_BACKUP_V2')
        try:
            MODEL_PATH.rename(backup)
            print(f"   Backed up old model → {backup.name}")
        except:
            pass
    
    if SHAP_PATH.exists():
        backup = SHAP_PATH.with_suffix('.pkl.OLD_BACKUP_V2')
        try:
            SHAP_PATH.rename(backup)
            print(f"   Backed up old explainer → {backup.name}")
        except:
            pass
    
    # Save new model
    joblib.dump(model, MODEL_PATH)
    print(f"   ✓ Saved {MODEL_PATH.name}")
    
    # Build SHAP explainer
    print(f"\n🧠 Building SHAP explainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, SHAP_PATH)
    print(f"   ✓ Saved {SHAP_PATH.name}")
    
    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("KERALA FLOOD RISK MODEL - TRAINING REPORT\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model:         Random Forest (300 trees, max_depth=12)\n")
        f.write(f"Training data: {len(X_train)} balanced samples\n")
        f.write(f"Features:      {len(FEATURE_COLS)}\n\n")
        
        f.write("PERFORMANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"Accuracy:    {metrics['accuracy']*100:.2f}%\n")
        f.write(f"Recall:      {metrics['recall']*100:.2f}%\n")
        f.write(f"Precision:   {metrics['precision']*100:.2f}%\n")
        f.write(f"F1 Score:    {metrics['f1']:.3f}\n")
        f.write(f"ROC-AUC:     {metrics['auc']:.3f}\n")
        f.write(f"CV Accuracy: {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%\n\n")
        
        f.write("FEATURE IMPORTANCE\n")
        f.write("-"*60 + "\n")
        for name, imp in metrics['importances']:
            f.write(f"{name:<28} {imp:.4f}\n")
    
    print(f"   ✓ Report saved: {REPORT_PATH.name}")


def main():
    print("\n" + "="*60)
    print("STEP 5: RETRAIN MODEL WITH NEW DATASET")
    print("="*60)
    
    # Load data
    X, y = load_data()
    
    # Train
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    
    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, X_train, y_train)
    
    # Sanity check
    sanity_passed = synthetic_sanity_check(model)
    
    # Real-world test
    real_world_test(model)
    
    # Save everything
    save_model_and_explainer(model, X_train, metrics)
    
    print("\n" + "="*60)
    if sanity_passed and metrics['accuracy'] > 0.80:
        print("🎉 SUCCESS! Model is ready for production.")
    else:
        print("⚠️ Model trained but may need more refinement.")
    print("="*60)
    print("\n👉 Next: Update Streamlit app to use 5 features")


if __name__ == "__main__":
    main()