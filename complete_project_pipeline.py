"""
Complete End-to-End Pipeline for Binary Prediction of Smoker Status Using Bio-Signals
Compatible with Google Colab
"""

# ============================================================================
# SETUP AND INSTALLATIONS (for Google Colab)
# ============================================================================
# Run these commands in Colab:
# !pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost catboost optuna shap
# !pip install imbalanced-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve, 
                             average_precision_score, accuracy_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, classification_report)
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings
import os
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
RANDOM_STATE = 42

# Create directories for saving outputs
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*80)
print("BINARY PREDICTION OF SMOKER STATUS USING BIO-SIGNALS")
print("="*80)

# Generate system architecture diagram
try:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    box_style = dict(boxstyle="round,pad=0.5", facecolor="lightblue", edgecolor="black", linewidth=2)
    arrow_style = dict(arrowstyle="->", lw=2, color="black")
    y_positions = [9, 7.5, 6, 4.5, 3, 1.5]
    boxes = [
        ("Data Loading\n(train.csv, test.csv)", 2.5, y_positions[0]),
        ("Data Preprocessing\n& Feature Engineering", 2.5, y_positions[1]),
        ("Feature Scaling\n(RobustScaler)", 2.5, y_positions[2]),
        ("Model Training\n(Baseline + Advanced)", 7.5, y_positions[3]),
        ("Hyperparameter\nOptimization (Optuna)", 7.5, y_positions[4]),
        ("Evaluation &\nVisualization", 7.5, y_positions[5]),
    ]
    for text, x, y in boxes:
        box = FancyBboxPatch((x-1.2, y-0.4), 2.4, 0.8, **box_style)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
    arrows = [
        ((2.5, y_positions[0]-0.4), (2.5, y_positions[1]+0.4)),
        ((2.5, y_positions[1]-0.4), (2.5, y_positions[2]+0.4)),
        ((2.5, y_positions[2]-0.4), (7.5, y_positions[3]+0.4)),
        ((7.5, y_positions[3]-0.4), (7.5, y_positions[4]+0.4)),
        ((7.5, y_positions[4]-0.4), (7.5, y_positions[5]+0.4)),
    ]
    for (start, end) in arrows:
        arrow = FancyArrowPatch(start, end, **arrow_style)
        ax.add_patch(arrow)
    ax.text(0.5, y_positions[1], "EDA\nAnalysis", ha='center', va='center', 
            fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    ax.text(0.5, y_positions[2], "PCA\nAnalysis", ha='center', va='center', 
            fontsize=9, bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))
    ax.text(10, y_positions[3], "LR, RF, SVM\nNB, KNN, AdaBoost\nLightGBM, XGBoost\nCatBoost, MLP", 
            ha='left', va='center', fontsize=8, 
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.text(6, 9.8, 'System Architecture: Complete ML Pipeline', 
            ha='center', va='top', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: figures/system_architecture.png")
except Exception as e:
    print(f"Note: Could not generate system architecture diagram: {e}")
    print("You can run generate_system_architecture.py separately if needed.")

# ============================================================================
# DATA LOADING
# ============================================================================
print("\n[1] Loading Data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Training columns: {list(train_df.columns)}")

# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n[2] Exploratory Data Analysis...")

# Basic statistics
print("\n--- Dataset Statistics ---")
print(train_df.describe())

# Check for missing values
print("\n--- Missing Values ---")
print(train_df.isnull().sum().sum())
print(test_df.isnull().sum().sum())

# Target distribution
target_dist = train_df['smoking'].value_counts()
print("\n--- Target Distribution ---")
print(target_dist)
print(f"Class balance: {target_dist[0] / len(train_df):.3f} (non-smoker) vs {target_dist[1] / len(train_df):.3f} (smoker)")

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='smoking', palette='viridis')
plt.title('Class Distribution: Smoking Status', fontsize=14, fontweight='bold')
plt.xlabel('Smoking Status (0=Non-smoker, 1=Smoker)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Non-smoker', 'Smoker'])
plt.tight_layout()
plt.savefig('figures/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/class_distribution.png")

# Correlation heatmap
plt.figure(figsize=(16, 14))
correlation_matrix = train_df.drop(['id'], axis=1).corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/correlation_heatmap.png")

# Feature distributions by target
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('id')
numeric_features.remove('smoking')

n_features = len(numeric_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for idx, feature in enumerate(numeric_features):
    ax = axes[idx]
    train_df.boxplot(column=feature, by='smoking', ax=ax, grid=False)
    ax.set_title(f'{feature}', fontsize=10)
    ax.set_xlabel('Smoking Status')
    ax.set_ylabel('Value')
    ax.get_figure().suptitle('')  # Remove default title

# Hide empty subplots
for idx in range(n_features, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Feature Distributions by Smoking Status', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/feature_distributions.png")

# ============================================================================
# DATA PREPROCESSING AND FEATURE ENGINEERING
# ============================================================================
print("\n[3] Data Preprocessing and Feature Engineering...")

# Separate features and target
X_train = train_df.drop(['id', 'smoking'], axis=1)
y_train = train_df['smoking']
X_test = test_df.drop(['id'], axis=1)
test_ids = test_df['id']

print(f"Training features shape: {X_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Feature engineering: Create additional features
def create_features(df):
    """Create engineered features"""
    df = df.copy()
    
    # BMI (Body Mass Index)
    df['BMI'] = df['weight(kg)'] / ((df['height(cm)'] / 100) ** 2)
    
    # Waist-to-height ratio
    df['waist_to_height'] = df['waist(cm)'] / df['height(cm)']
    
    # Average eyesight
    df['avg_eyesight'] = (df['eyesight(left)'] + df['eyesight(right)']) / 2
    
    # Average hearing
    df['avg_hearing'] = (df['hearing(left)'] + df['hearing(right)']) / 2
    
    # Blood pressure ratio
    df['bp_ratio'] = df['systolic'] / (df['relaxation'] + 1e-6)
    
    # Cholesterol ratios
    df['total_chol_hdl'] = df['Cholesterol'] / (df['HDL'] + 1e-6)
    df['ldl_hdl_ratio'] = df['LDL'] / (df['HDL'] + 1e-6)
    df['trig_hdl_ratio'] = df['triglyceride'] / (df['HDL'] + 1e-6)
    
    # Liver function indicators
    df['ast_alt_ratio'] = df['AST'] / (df['ALT'] + 1e-6)
    
    # Metabolic indicators
    df['metabolic_risk'] = (df['fasting blood sugar'] + df['triglyceride']) / (df['HDL'] + 1e-6)
    
    return df

X_train_eng = create_features(X_train)
X_test_eng = create_features(X_test)

print(f"Features after engineering: {X_train_eng.shape[1]}")

# Scaling
scaler = RobustScaler()  # Robust to outliers
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Convert back to DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_eng.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_eng.columns)

print("Preprocessing completed!")

# ============================================================================
# PCA ANALYSIS AND VISUALIZATION
# ============================================================================
print("\n[4] Principal Component Analysis...")

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# PCA visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, 
                     cmap='viridis', alpha=0.6, s=20)
plt.colorbar(scatter, label='Smoking Status')
plt.xlabel(f'First Principal Component (Explained Variance: {pca.explained_variance_ratio_[0]:.2%})', 
           fontsize=12)
plt.ylabel(f'Second Principal Component (Explained Variance: {pca.explained_variance_ratio_[1]:.2%})', 
           fontsize=12)
plt.title('PCA Visualization: First Two Principal Components', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pca_scatterplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/pca_scatterplot.png")

# Explained variance plot
pca_full = PCA()
pca_full.fit(X_train_scaled)
explained_var = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, min(21, len(explained_var)+1)), explained_var[:20], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
plt.title('PCA: Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('figures/pca_explained_variance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/pca_explained_variance.png")

print(f"First 2 PCs explain {explained_var[1]:.2%} of variance")
print(f"First 10 PCs explain {explained_var[9]:.2%} of variance")

# ============================================================================
# BASELINE MODELS
# ============================================================================
print("\n[5] Training Baseline Models...")

# Setup for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
    'SVM': SVC(probability=True, random_state=RANDOM_STATE),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'AdaBoost': AdaBoostClassifier(random_state=RANDOM_STATE)
}

baseline_results = {}

for name, model in baseline_models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
    
    # Train on full data
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_pred = model.predict(X_train_scaled)
    
    # Metrics
    roc_auc = roc_auc_score(y_train, y_pred_proba)
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)
    ap_score = average_precision_score(y_train, y_pred_proba)
    
    baseline_results[name] = {
        'model': model,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap_score': ap_score,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  ROC-AUC: {roc_auc:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

# ============================================================================
# ADVANCED MODELS
# ============================================================================
print("\n[6] Training Advanced Models...")

advanced_models = {}

# LightGBM
print("\nTraining LightGBM...")
lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train_scaled, y_train)
y_pred_proba_lgb = lgb_model.predict_proba(X_train_scaled)[:, 1]
y_pred_lgb = lgb_model.predict(X_train_scaled)

lgb_metrics = {
    'roc_auc': roc_auc_score(y_train, y_pred_proba_lgb),
    'accuracy': accuracy_score(y_train, y_pred_lgb),
    'precision': precision_score(y_train, y_pred_lgb),
    'recall': recall_score(y_train, y_pred_lgb),
    'f1': f1_score(y_train, y_pred_lgb),
    'ap_score': average_precision_score(y_train, y_pred_proba_lgb)
}

cv_scores_lgb = cross_val_score(lgb_model, X_train_scaled, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
lgb_metrics['cv_mean'] = cv_scores_lgb.mean()
lgb_metrics['cv_std'] = cv_scores_lgb.std()
lgb_metrics['model'] = lgb_model
lgb_metrics['y_pred_proba'] = y_pred_proba_lgb

advanced_models['LightGBM'] = lgb_metrics
print(f"  CV ROC-AUC: {cv_scores_lgb.mean():.4f} (+/- {cv_scores_lgb.std()*2:.4f})")
print(f"  ROC-AUC: {lgb_metrics['roc_auc']:.4f}")

# XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)
y_pred_proba_xgb = xgb_model.predict_proba(X_train_scaled)[:, 1]
y_pred_xgb = xgb_model.predict(X_train_scaled)

xgb_metrics = {
    'roc_auc': roc_auc_score(y_train, y_pred_proba_xgb),
    'accuracy': accuracy_score(y_train, y_pred_xgb),
    'precision': precision_score(y_train, y_pred_xgb),
    'recall': recall_score(y_train, y_pred_xgb),
    'f1': f1_score(y_train, y_pred_xgb),
    'ap_score': average_precision_score(y_train, y_pred_proba_xgb)
}

cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
xgb_metrics['cv_mean'] = cv_scores_xgb.mean()
xgb_metrics['cv_std'] = cv_scores_xgb.std()
xgb_metrics['model'] = xgb_model
xgb_metrics['y_pred_proba'] = y_pred_proba_xgb

advanced_models['XGBoost'] = xgb_metrics
print(f"  CV ROC-AUC: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std()*2:.4f})")
print(f"  ROC-AUC: {xgb_metrics['roc_auc']:.4f}")

# CatBoost
print("\nTraining CatBoost...")
cat_model = cb.CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=7,
    random_state=RANDOM_STATE,
    verbose=False
)
cat_model.fit(X_train_scaled, y_train)
y_pred_proba_cat = cat_model.predict_proba(X_train_scaled)[:, 1]
y_pred_cat = cat_model.predict(X_train_scaled)

cat_metrics = {
    'roc_auc': roc_auc_score(y_train, y_pred_proba_cat),
    'accuracy': accuracy_score(y_train, y_pred_cat),
    'precision': precision_score(y_train, y_pred_cat),
    'recall': recall_score(y_train, y_pred_cat),
    'f1': f1_score(y_train, y_pred_cat),
    'ap_score': average_precision_score(y_train, y_pred_proba_cat)
}

cv_scores_cat = cross_val_score(cat_model, X_train_scaled, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
cat_metrics['cv_mean'] = cv_scores_cat.mean()
cat_metrics['cv_std'] = cv_scores_cat.std()
cat_metrics['model'] = cat_model
cat_metrics['y_pred_proba'] = y_pred_proba_cat

advanced_models['CatBoost'] = cat_metrics
print(f"  CV ROC-AUC: {cv_scores_cat.mean():.4f} (+/- {cv_scores_cat.std()*2:.4f})")
print(f"  ROC-AUC: {cat_metrics['roc_auc']:.4f}")

# MLP (Neural Network)
print("\nTraining MLP...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=256,
    learning_rate='adaptive',
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=True,
    validation_fraction=0.1
)
mlp_model.fit(X_train_scaled, y_train)
y_pred_proba_mlp = mlp_model.predict_proba(X_train_scaled)[:, 1]
y_pred_mlp = mlp_model.predict(X_train_scaled)

mlp_metrics = {
    'roc_auc': roc_auc_score(y_train, y_pred_proba_mlp),
    'accuracy': accuracy_score(y_train, y_pred_mlp),
    'precision': precision_score(y_train, y_pred_mlp),
    'recall': recall_score(y_train, y_pred_mlp),
    'f1': f1_score(y_train, y_pred_mlp),
    'ap_score': average_precision_score(y_train, y_pred_proba_mlp)
}

cv_scores_mlp = cross_val_score(mlp_model, X_train_scaled, y_train, cv=cv, 
                                scoring='roc_auc', n_jobs=-1)
mlp_metrics['cv_mean'] = cv_scores_mlp.mean()
mlp_metrics['cv_std'] = cv_scores_mlp.std()
mlp_metrics['model'] = mlp_model
mlp_metrics['y_pred_proba'] = y_pred_proba_mlp

advanced_models['MLP'] = mlp_metrics
print(f"  CV ROC-AUC: {cv_scores_mlp.mean():.4f} (+/- {cv_scores_mlp.std()*2:.4f})")
print(f"  ROC-AUC: {mlp_metrics['roc_auc']:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING (Optuna for CatBoost - Best Model)
# ============================================================================
print("\n[7] Hyperparameter Tuning with Optuna (CatBoost)...")

def objective(trial):
    """Optuna objective function for CatBoost"""
    params = {
        'iterations': trial.suggest_int('iterations', 300, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_strength': trial.suggest_float('random_strength', 0, 1),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_state': RANDOM_STATE,
        'verbose': False
    }
    
    model = cb.CatBoostClassifier(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, 
                            scoring='roc_auc', n_jobs=-1)
    return scores.mean()

# Run Optuna optimization
study = optuna.create_study(direction='maximize', study_name='catboost_optimization')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"\nBest parameters: {study.best_params}")
print(f"Best CV score: {study.best_value:.4f}")

# Train best model
best_cat_model = cb.CatBoostClassifier(**study.best_params, random_state=RANDOM_STATE, verbose=False)
best_cat_model.fit(X_train_scaled, y_train)
y_pred_proba_best = best_cat_model.predict_proba(X_train_scaled)[:, 1]
y_pred_best = best_cat_model.predict(X_train_scaled)

best_cat_metrics = {
    'roc_auc': roc_auc_score(y_train, y_pred_proba_best),
    'accuracy': accuracy_score(y_train, y_pred_best),
    'precision': precision_score(y_train, y_pred_best),
    'recall': recall_score(y_train, y_pred_best),
    'f1': f1_score(y_train, y_pred_best),
    'ap_score': average_precision_score(y_train, y_pred_proba_best),
    'model': best_cat_model,
    'y_pred_proba': y_pred_proba_best
}

cv_scores_best = cross_val_score(best_cat_model, X_train_scaled, y_train, cv=cv, 
                                 scoring='roc_auc', n_jobs=-1)
best_cat_metrics['cv_mean'] = cv_scores_best.mean()
best_cat_metrics['cv_std'] = cv_scores_best.std()

advanced_models['CatBoost (Tuned)'] = best_cat_metrics
print(f"Tuned CatBoost - CV ROC-AUC: {cv_scores_best.mean():.4f} (+/- {cv_scores_best.std()*2:.4f})")
print(f"Tuned CatBoost - ROC-AUC: {best_cat_metrics['roc_auc']:.4f}")

# ============================================================================
# OUT-OF-FOLD PREDICTIONS (K-Fold CV)
# ============================================================================
print("\n[8] Generating Out-of-Fold Predictions...")

# Use best model for OOF predictions
oof_predictions = np.zeros(len(X_train_scaled))
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train)):
    print(f"  Fold {fold+1}/{n_splits}...")
    X_fold_train, X_fold_val = X_train_scaled.iloc[train_idx], X_train_scaled.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    fold_model = cb.CatBoostClassifier(**study.best_params, random_state=RANDOM_STATE, verbose=False)
    fold_model.fit(X_fold_train, y_fold_train)
    oof_predictions[val_idx] = fold_model.predict_proba(X_fold_val)[:, 1]

oof_roc_auc = roc_auc_score(y_train, oof_predictions)
print(f"OOF ROC-AUC: {oof_roc_auc:.4f}")

# ============================================================================
# MODEL COMPARISON AND VISUALIZATIONS
# ============================================================================
print("\n[9] Generating Model Comparison Visualizations...")

# Combine all results
all_results = {**baseline_results, **advanced_models}

# Create comparison DataFrame
comparison_data = []
for name, results in all_results.items():
    comparison_data.append({
        'Model': name,
        'CV ROC-AUC Mean': results['cv_mean'],
        'CV ROC-AUC Std': results['cv_std'],
        'ROC-AUC': results['roc_auc'],
        'Accuracy': results['accuracy'],
        'Precision': results['precision'],
        'Recall': results['recall'],
        'F1-Score': results['f1'],
        'AP Score': results['ap_score']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('ROC-AUC', ascending=False)
comparison_df.to_csv('results/model_comparison.csv', index=False)
print("\nModel Comparison Table:")
print(comparison_df.to_string(index=False))

# ROC Curves
plt.figure(figsize=(12, 8))
for name, results in all_results.items():
    fpr, tpr, _ = roc_curve(y_train, results['y_pred_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={results['roc_auc']:.4f})", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves: Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/roc_curves.png")

# Precision-Recall Curves
plt.figure(figsize=(12, 8))
for name, results in all_results.items():
    precision, recall, _ = precision_recall_curve(y_train, results['y_pred_proba'])
    plt.plot(recall, precision, label=f"{name} (AP={results['ap_score']:.4f})", linewidth=2)

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves: Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/pr_curves.png")

# Model Performance Bar Chart
plt.figure(figsize=(14, 8))
x_pos = np.arange(len(comparison_df))
width = 0.15

metrics_to_plot = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, metric in enumerate(metrics_to_plot):
    plt.bar(x_pos + idx*width, comparison_df[metric], width, 
            label=metric, color=colors[idx], alpha=0.8)

plt.xlabel('Models', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xticks(x_pos + width*2, comparison_df['Model'], rotation=45, ha='right')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('figures/model_comparison_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/model_comparison_bars.png")

# Feature Importance (for tree-based models)
best_model = best_cat_model
feature_importance = best_model.get_feature_importance()
feature_names = X_train_scaled.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Top 20 features
top_features = importance_df.head(20)

plt.figure(figsize=(12, 8))
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 20 Feature Importance (CatBoost)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('figures/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/feature_importance.png")

importance_df.to_csv('results/feature_importance.csv', index=False)

# ============================================================================
# ABLATION STUDY
# ============================================================================
print("\n[10] Ablation Study...")

ablation_results = []

# Baseline: No feature engineering
print("  Testing: No feature engineering...")
X_train_base = scaler.fit_transform(X_train)
cat_base = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=7, 
                                  random_state=RANDOM_STATE, verbose=False)
cv_scores_base = cross_val_score(cat_base, X_train_base, y_train, cv=cv, 
                                 scoring='roc_auc', n_jobs=-1)
ablation_results.append({
    'Configuration': 'No Feature Engineering',
    'CV ROC-AUC': cv_scores_base.mean(),
    'CV Std': cv_scores_base.std()
})

# With feature engineering
ablation_results.append({
    'Configuration': 'With Feature Engineering',
    'CV ROC-AUC': best_cat_metrics['cv_mean'],
    'CV Std': best_cat_metrics['cv_std']
})

# Different scalers
for scaler_name, scaler_obj in [('StandardScaler', StandardScaler()), 
                                ('RobustScaler', RobustScaler())]:
    print(f"  Testing: {scaler_name}...")
    X_scaled = scaler_obj.fit_transform(X_train_eng)
    cat_scaled = cb.CatBoostClassifier(iterations=300, learning_rate=0.05, depth=7, 
                                       random_state=RANDOM_STATE, verbose=False)
    cv_scores_scaled = cross_val_score(cat_scaled, X_scaled, y_train, cv=cv, 
                                       scoring='roc_auc', n_jobs=-1)
    ablation_results.append({
        'Configuration': f'{scaler_name}',
        'CV ROC-AUC': cv_scores_scaled.mean(),
        'CV Std': cv_scores_scaled.std()
    })

ablation_df = pd.DataFrame(ablation_results)
ablation_df = ablation_df.sort_values('CV ROC-AUC', ascending=False)
ablation_df.to_csv('results/ablation_study.csv', index=False)

print("\nAblation Study Results:")
print(ablation_df.to_string(index=False))

# Ablation visualization
plt.figure(figsize=(10, 6))
y_pos = np.arange(len(ablation_df))
plt.barh(y_pos, ablation_df['CV ROC-AUC'], xerr=ablation_df['CV Std'], 
         color='coral', alpha=0.8, capsize=5)
plt.yticks(y_pos, ablation_df['Configuration'])
plt.xlabel('CV ROC-AUC Score', fontsize=12)
plt.title('Ablation Study: Impact of Different Configurations', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('figures/ablation_study.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: figures/ablation_study.png")

# ============================================================================
# FINAL PREDICTIONS
# ============================================================================
print("\n[11] Generating Final Predictions...")

# Use best model for test predictions
final_predictions = best_cat_model.predict_proba(X_test_scaled)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'smoking': final_predictions
})
submission.to_csv('results/final_submission.csv', index=False)
print("Saved: results/final_submission.csv")
print(f"Prediction range: [{final_predictions.min():.4f}, {final_predictions.max():.4f}]")

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n[12] Saving Models...")

with open('models/best_catboost_model.pkl', 'wb') as f:
    pickle.dump(best_cat_model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models saved!")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("PROJECT SUMMARY")
print("="*80)
print(f"\nBest Model: CatBoost (Tuned)")
print(f"  CV ROC-AUC: {best_cat_metrics['cv_mean']:.4f} (+/- {best_cat_metrics['cv_std']*2:.4f})")
print(f"  ROC-AUC: {best_cat_metrics['roc_auc']:.4f}")
print(f"  Accuracy: {best_cat_metrics['accuracy']:.4f}")
print(f"  Precision: {best_cat_metrics['precision']:.4f}")
print(f"  Recall: {best_cat_metrics['recall']:.4f}")
print(f"  F1-Score: {best_cat_metrics['f1']:.4f}")
print(f"  AP Score: {best_cat_metrics['ap_score']:.4f}")

print(f"\nTotal Features: {X_train_scaled.shape[1]}")
print(f"Training Samples: {len(X_train_scaled)}")
print(f"Test Samples: {len(X_test_scaled)}")

print("\nAll visualizations and results saved in 'figures/' and 'results/' directories")
print("="*80)

