"""
FINAL OPTIMIZED CODE FOR GOOGLE COLAB
Generates all 4 required figures for the LaTeX report
Optimized for speed - runs in 30-60 minutes on Colab free tier
"""

# ============================================================================
# CELL 1: INSTALL DEPENDENCIES (Run this first)
# ============================================================================
# !pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost catboost -q

# ============================================================================
# CELL 2: IMPORTS AND SETUP
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score)
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('default')

# Set random seeds
np.random.seed(42)
RANDOM_STATE = 42

# Create directories
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*80)
print("FINAL FIGURE GENERATION - OPTIMIZED FOR SPEED")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# ============================================================================
# CELL 3: DATA LOADING AND PREPROCESSING (OPTIMIZED)
# ============================================================================
print("\n[1] Loading and Preprocessing Data...")

# Load data
train_df = pd.read_csv('train.csv')
print(f"Original training set shape: {train_df.shape}")

# OPTIMIZATION: Use 15% subsample for speed (still ~24K samples - plenty for good results)
sample_size = int(0.15 * len(train_df))
train_df = train_df.sample(n=sample_size, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Subsampled training set shape: {train_df.shape} (15% for speed)")

# Separate features and target
X_train = train_df.drop(['id', 'smoking'], axis=1)
y_train = train_df['smoking']

# Feature engineering
def create_features(df):
    """Create engineered features"""
    df = df.copy()
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    df['waist_to_height'] = df['waist(cm)'] / df['height(cm)']
    df['avg_eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
    df['avg_hearing'] = (df['hearing(left)'] + df['hearing(right)']) / 2
    df['bp_ratio'] = df['systolic'] / (df['relaxation'] + 1e-6)
    df['total_chol_hdl'] = df['Cholesterol'] / (df['HDL'] + 1e-6)
    df['ldl_hdl_ratio'] = df['LDL'] / (df['HDL'] + 1e-6)
    df['trig_hdl_ratio'] = df['triglyceride'] / (df['HDL'] + 1e-6)
    df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-6)
    df['metabolic_risk'] = (df['fasting blood sugar'] + df['triglyceride']) / (df['HDL'] + 1e-6)
    return df

X_train_eng = create_features(X_train)
print(f"Features after engineering: {X_train_eng.shape[1]}")

# Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_eng.columns)

print("✓ Preprocessing completed!")

# ============================================================================
# CELL 4: QUICK MODEL TRAINING (OPTIMIZED - SKIP SVM)
# ============================================================================
print("\n[2] Training Models (Optimized - Fast Training Mode)...")

# OPTIMIZATION: Use 3-fold CV instead of 5-fold (faster, still reliable)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

all_results = {}

# Baseline Models (Quick training - reduced iterations)
print("\nTraining baseline models...")
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=300, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
    # SKIP SVM - too slow
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=RANDOM_STATE)
}

for name, model in baseline_models.items():
    print(f"  {name}...", end=' ', flush=True)
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    all_results[name] = {
        'roc_auc': roc_auc_score(y_train, y_pred_proba),
        'ap_score': average_precision_score(y_train, y_pred_proba),
        'y_pred_proba': y_pred_proba
    }
    print(f"ROC-AUC: {all_results[name]['roc_auc']:.4f}")

# Advanced Models (Reduced iterations for speed)
print("\nTraining advanced models...")
advanced_models = {
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                  random_state=RANDOM_STATE, n_jobs=-1, eval_metric='logloss', verbosity=0),
    'CatBoost': cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, 
                                       random_state=RANDOM_STATE, verbose=False),
    'MLP': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=RANDOM_STATE, 
                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)
}

for name, model in advanced_models.items():
    print(f"  {name}...", end=' ', flush=True)
    model.fit(X_train_scaled, y_train)
    y_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    
    all_results[name] = {
        'roc_auc': roc_auc_score(y_train, y_pred_proba),
        'ap_score': average_precision_score(y_train, y_pred_proba),
        'y_pred_proba': y_pred_proba
    }
    print(f"ROC-AUC: {all_results[name]['roc_auc']:.4f}")

# Best model (CatBoost with slightly better params for feature importance)
print("\nTraining best CatBoost model for feature importance...")
best_cat_model = cb.CatBoostClassifier(
    iterations=150, learning_rate=0.05, depth=7, 
    random_state=RANDOM_STATE, verbose=False
)
best_cat_model.fit(X_train_scaled, y_train)
y_pred_proba_best = best_cat_model.predict_proba(X_train_scaled)[:, 1]

all_results['CatBoost (Tuned)'] = {
    'roc_auc': roc_auc_score(y_train, y_pred_proba_best),
    'ap_score': average_precision_score(y_train, y_pred_proba_best),
    'y_pred_proba': y_pred_proba_best
}
print(f"CatBoost (Tuned) - ROC-AUC: {all_results['CatBoost (Tuned)']['roc_auc']:.4f}")

print("\n✓ All models trained!")

# ============================================================================
# CELL 5: GENERATE FIGURE 1 - ROC CURVES
# ============================================================================
print("\n[3] Generating ROC Curves...")

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

for idx, (name, results) in enumerate(all_results.items()):
    fpr, tpr, _ = roc_curve(y_train, results['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={results['roc_auc']:.4f})", 
             linewidth=2.5, color=colors[idx])

plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('ROC Curves: Model Comparison', fontsize=15, fontweight='bold', pad=15)
plt.legend(loc='lower right', fontsize=9, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: figures/roc_curves.png")

# ============================================================================
# CELL 6: GENERATE FIGURE 2 - PRECISION-RECALL CURVES
# ============================================================================
print("\n[4] Generating Precision-Recall Curves...")

plt.figure(figsize=(10, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

for idx, (name, results) in enumerate(all_results.items()):
    precision, recall, _ = precision_recall_curve(y_train, results['y_pred_proba'])
    plt.plot(recall, precision, label=f"{name} (AP={results['ap_score']:.4f})", 
             linewidth=2.5, color=colors[idx])

plt.xlabel('Recall', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')
plt.title('Precision-Recall Curves: Model Comparison', fontsize=15, fontweight='bold', pad=15)
plt.legend(loc='lower left', fontsize=9, framealpha=0.9)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.tight_layout()
plt.savefig('figures/pr_curves.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: figures/pr_curves.png")

# ============================================================================
# CELL 7: GENERATE FIGURE 3 - FEATURE IMPORTANCE
# ============================================================================
print("\n[5] Generating Feature Importance Plot...")

feature_importance = best_cat_model.get_feature_importance()
feature_names = X_train_scaled.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

top_features = importance_df.head(20)

plt.figure(figsize=(10, 8))
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
bars = plt.barh(range(len(top_features)), top_features['importance'], 
                color=colors_bar, alpha=0.8, edgecolor='black', linewidth=0.5)
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
plt.xlabel('Feature Importance', fontsize=13, fontweight='bold')
plt.title('Top 20 Feature Importance (CatBoost)', fontsize=15, fontweight='bold', pad=15)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: figures/feature_importance.png")

# ============================================================================
# CELL 8: GENERATE FIGURE 4 - ABLATION STUDY
# ============================================================================
print("\n[6] Generating Ablation Study Plot...")

ablation_results = []

# No feature engineering
print("  Testing: No feature engineering...")
X_train_base = scaler.fit_transform(X_train)
cat_base = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, 
                                  random_state=RANDOM_STATE, verbose=False)
cv_scores_base = cross_val_score(cat_base, X_train_base, y_train, cv=cv, 
                                 scoring='roc_auc', n_jobs=-1)
ablation_results.append({
    'Configuration': 'No Feature Engineering',
    'CV ROC-AUC': cv_scores_base.mean(),
    'CV Std': cv_scores_base.std()
})

# With feature engineering
print("  Testing: With feature engineering...")
cv_scores_eng = cross_val_score(best_cat_model, X_train_scaled, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
ablation_results.append({
    'Configuration': 'With Feature Engineering',
    'CV ROC-AUC': cv_scores_eng.mean(),
    'CV Std': cv_scores_eng.std()
})

# Different scalers
print("  Testing: StandardScaler...")
X_scaled_std = StandardScaler().fit_transform(X_train_eng)
cat_std = cb.CatBoostClassifier(iterations=100, learning_rate=0.1, depth=5, 
                                 random_state=RANDOM_STATE, verbose=False)
cv_scores_std = cross_val_score(cat_std, X_scaled_std, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
ablation_results.append({
    'Configuration': 'StandardScaler',
    'CV ROC-AUC': cv_scores_std.mean(),
    'CV Std': cv_scores_std.std()
})

# RobustScaler (already done)
ablation_results.append({
    'Configuration': 'RobustScaler',
    'CV ROC-AUC': cv_scores_eng.mean(),
    'CV Std': cv_scores_eng.std()
})

ablation_df = pd.DataFrame(ablation_results)
ablation_df = ablation_df.sort_values('CV ROC-AUC', ascending=False)

# Create visualization
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(ablation_df))
colors_ablation = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
bars = plt.barh(y_pos, ablation_df['CV ROC-AUC'], xerr=ablation_df['CV Std'], 
                color=colors_ablation[:len(ablation_df)], alpha=0.8, capsize=5, 
                edgecolor='black', linewidth=1.5)

# Add value labels
for i, (idx, row) in enumerate(ablation_df.iterrows()):
    plt.text(row['CV ROC-AUC'] + row['CV Std'] + 0.0005, i, f"{row['CV ROC-AUC']:.4f}", 
             va='center', fontsize=10, fontweight='bold')

plt.yticks(y_pos, ablation_df['Configuration'], fontsize=11)
plt.xlabel('CV ROC-AUC Score', fontsize=13, fontweight='bold')
plt.title('Ablation Study: Impact of Different Configurations', fontsize=15, fontweight='bold', pad=15)
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
plt.xlim([ablation_df['CV ROC-AUC'].min() - 0.01, ablation_df['CV ROC-AUC'].max() + 0.015])
plt.tight_layout()
plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: figures/ablation_study.png")

# ============================================================================
# CELL 9: VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("FIGURE GENERATION COMPLETE")
print("="*80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nGenerated Figures:")
required_files = ['roc_curves.png', 'pr_curves.png', 'feature_importance.png', 'ablation_study.png']
all_exist = True
for f in required_files:
    path = f'figures/{f}'
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  ✓ {f:30s} {size_kb:6.1f} KB")
    else:
        print(f"  ✗ {f:30s} MISSING")
        all_exist = False

if all_exist:
    print("\n✓ All 4 figures generated successfully!")
    print("✓ Ready for LaTeX inclusion")
    print("\nTo download from Colab:")
    print("  1. Click folder icon (left sidebar)")
    print("  2. Navigate to 'figures' folder")
    print("  3. Right-click each PNG file → Download")
    print("  4. Or zip the folder: !zip -r figures.zip figures/")
else:
    print("\n✗ Some figures are missing - check errors above")
print("="*80)

