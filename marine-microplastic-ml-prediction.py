# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Loading the  dataset


dataset = pd.read_csv('Dataset.csv')
print("Dataset shape:", dataset.shape)

# Convert month into cyclic features.
# Month 12 and month 1 are close in time
# so sine/cosine encoding preserves this cyclic relationship


dataset['month_sin'] = np.sin(2 * np.pi * dataset['month'] / 12)
dataset['month_cos'] = np.cos(2 * np.pi * dataset['month'] / 12)

# Target variable (continuous concentration values)

target_col = 'Level 0 (pieces/m3)'
y = dataset[target_col].copy()   # continuous values

# Select only the predictor variables used for training


feature_cols = [
    'windspeed (m/s)',
    'significant wave height (m)',
    'longitude (degree: E+, W-)',
    'latitude (degree: N+, S-)',
    'month_sin',
    'month_cos'
]
X = dataset[feature_cols].copy()

# Drop rows with missing values in features or target


data_clean = pd.concat([X, y], axis=1).dropna()
X = data_clean[feature_cols]
y = data_clean[target_col]

print("Total samples:", len(X))
print("Mean:", y.mean())
print("Median:", y.median())
print("Standard deviation:", y.std())

# Split the data by year
# years before 2010  for training and after 2010  for testing


dataset_split = dataset.loc[data_clean.index].copy()
dataset_split['target'] = y
train_mask = dataset_split['year'] <= 2010
test_mask = dataset_split['year'] > 2010

X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

print(f"\nTraining samples (year ≤ 2010): {len(X_train)}")
print(f"Testing samples (year > 2010): {len(X_test)}")

# If the year based split leaves too few test samples
# switch to a normal 80/20 split
# For future use with different dataset


if len(X_test) < 10:
    print("\nWarning: Not enough test samples by year split. Using random split (80/20).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Standardize the features before training
# This mainly helps distance based models like KNN

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Helper functions
# Functions used to calculate evaluation metrics and display the results


threshold = 0.1
# Threshold used to convert regression predictions into low/high concentration classes


def print_regression_metrics(model_name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} Regression Results")
    print(f"MAE : {mae:.4f} pieces/m3")
    print(f"RMSE: {rmse:.4f} pieces/m3")
    print(f"R2  : {r2:.4f}")

    results = {'Model': model_name}
    results['MAE'] = round(mae, 4)
    results['RMSE'] = round(rmse, 4)
    results['R2'] = round(r2, 3)
    return results


# Convert the regression output into binary classes
# so the predictions can also be evaluated as low/high concentration


def print_classification_metrics(model_name, y_true, y_pred):
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)



    print(f"\nClassification Metrics (threshold = {threshold})")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    return {'Model': model_name, 'Acc': round(accuracy, 4), 'Prec': round(precision, 4),
            'Rec': round(recall, 4), 'F1': round(f1, 4)}


# Plot the confusion matrix after applying
# the classification threshold


def plot_confusion_matrix_regression(model_name, y_true, y_pred):
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    cm = confusion_matrix(y_true_binary, y_pred_binary)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low (<0.1)', 'High (>=0.1)'],
                yticklabels=['Low (<0.1)', 'High (>=0.1)'])
    plt.title(model_name + " Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# Store the evaluation results from all models

all_reg_results = []
all_clf_results = []

#   MODEL 1: XGBOOST REGRESSOR
# Train and evaluate the XGBoost regressor

print("\n\n  XGBoost Regressor")

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5,
                         subsample=0.8, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
reg_metrics = print_regression_metrics("XGBoost", y_test, y_pred_xgb)
clf_metrics = print_classification_metrics("XGBoost", y_test, y_pred_xgb)
plot_confusion_matrix_regression("XGBoost", y_test, y_pred_xgb)
all_reg_results.append(reg_metrics)
all_clf_results.append(clf_metrics)

#   MODEL 2: RANDOM FOREST REGRESSOR
# Train and evaluate the RF regressor

print("\n\n Random Forest Regressor")
rf_model = RandomForestRegressor(n_estimators=200, max_depth=None,
                                 min_samples_split=2, min_samples_leaf=1,
                                 random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
reg_metrics = print_regression_metrics("Random Forest", y_test, y_pred_rf)
clf_metrics = print_classification_metrics("Random Forest", y_test, y_pred_rf)
plot_confusion_matrix_regression("Random Forest", y_test, y_pred_rf)
all_reg_results.append(reg_metrics)
all_clf_results.append(clf_metrics)

#   MODEL 3: KNN REGRESSOR
# Train and evaluate the KNN regressor

print("\n K Nearest Neighbors Regressor")

knn_model = KNeighborsRegressor(n_neighbors=7, weights='distance', metric='euclidean', n_jobs=-1)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
reg_metrics = print_regression_metrics("KNN", y_test, y_pred_knn)
clf_metrics = print_classification_metrics("KNN", y_test, y_pred_knn)
plot_confusion_matrix_regression("KNN", y_test, y_pred_knn)
all_reg_results.append(reg_metrics)
all_clf_results.append(clf_metrics)

# MODEL 4: Train and evaluate the stacking ensemble
# Random Forest, XGBoost and KNN are used as base learners
# Linear Regression as Meta Learner




print("\n Stacking Regressor")

base_learners = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)),
    ('knn', KNeighborsRegressor(n_neighbors=7, weights='distance', n_jobs=-1))
]
meta_learner = LinearRegression()
stack_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner,
                                cv=5, n_jobs=-1)
stack_model.fit(X_train_scaled, y_train)
y_pred_stack = stack_model.predict(X_test_scaled)
reg_metrics = print_regression_metrics("Stacking Ensemble", y_test, y_pred_stack)
clf_metrics = print_classification_metrics("Stacking Ensemble", y_test, y_pred_stack)
plot_confusion_matrix_regression("Stacking Ensemble", y_test, y_pred_stack)
all_reg_results.append(reg_metrics)
all_clf_results.append(clf_metrics)

# FInal Comparison Table (Regression and Classification)
# Create tables to compare the performance of all trained models

print("\n\nFINAL MODEL COMPARISON – REGRESSION METRICS")
reg_comparison = pd.DataFrame(all_reg_results)
print(reg_comparison.to_string(index=False))

print("\n\n FINAL MODEL COMPARISON – CLASSIFICATION METRICS (threshold=0.1)")
clf_comparison = pd.DataFrame(all_clf_results)
print(clf_comparison.to_string(index=False))

# Bar charts for regression metrics
reg_metrics_names = ['MAE', 'RMSE', 'R2']  # note: lower MAE/RMSE better, higher R2 better
model_names = ['XGBoost', 'Random Forest', 'KNN', 'Stacking']
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Regression Performance Comparison', fontsize=16, fontweight='bold')
for i, metric in enumerate(reg_metrics_names):
    values = reg_comparison[metric].values
    bars = axes[i].bar(model_names, values, color=colors, edgecolor='white', width=0.6)
    axes[i].set_title(metric, fontsize=13, fontweight='bold')
    if metric in ['MAE', 'RMSE']:
        axes[i].set_ylabel('Error (pieces/m³)')
    else:
        axes[i].set_ylim(0, 1.05)
        axes[i].set_ylabel('R² (higher better)')
    axes[i].tick_params(axis='x', rotation=15)
    axes[i].grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if metric=='R2' else 0.02),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# Bar chart for classification F1 score

fig, ax = plt.subplots(figsize=(8,5))
f1_values = clf_comparison['F1'].values
bars = ax.bar(model_names, f1_values, color=colors, edgecolor='white', width=0.6)
ax.set_title('Classification F1 Score (threshold=0.1)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_ylabel('F1 Score')
ax.tick_params(axis='x', rotation=15)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, f1_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

