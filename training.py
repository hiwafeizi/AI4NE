import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress LightGBM warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Step 1: Load the preprocessed training and testing datasets
print("Loading preprocessed datasets...")
X_train = pd.read_csv("X_train_processed.csv")
X_test = pd.read_csv("X_test_processed.csv")
y_train = pd.read_csv("y_train_processed.csv").values.ravel()
y_test = pd.read_csv("y_test_processed.csv").values.ravel()

# Validate data shapes
assert X_train.shape[0] == y_train.shape[0], "Mismatch in training data and labels"
assert X_test.shape[0] == y_test.shape[0], "Mismatch in test data and labels"

# Step 2: Define the parameter grid for Grid Search
print("\nDefining the parameter grid for Grid Search...")
param_grid = {
    'num_leaves': [20, 50, 100],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [-1, 5, 8],
    'n_estimators': [20, 50, 100],
    'feature_fraction': [0.2, 0.6, 0.8]
}

# Step 3: Perform Grid Search for Hyperparameter Tuning
print("\nPerforming Grid Search for Hyperparameter Tuning...")
grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        class_weight='balanced',
        verbose=-1
    ),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Step 4: Get the best parameters and model
print("\nBest parameters found by Grid Search:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

# Step 5: Train the final model with best parameters
print("\nTraining the best model on the full training data...")
best_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
)
print("Model training completed.")

# Step 6: Predict on the test set
print("\nPredicting on the test set...")
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Step 7: Evaluate the model
print("\nEvaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 9: Save the trained model
print("\nSaving the trained model...")
import joblib
joblib.dump(best_model, "lightgbm_fire_model_optimized.pkl")
print("Trained model saved as 'lightgbm_fire_model_optimized.pkl'.")
