import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb


# Load the trained model and test data
print("Loading the trained model and test data...")
best_model = joblib.load("lightgbm_fire_model_optimized.pkl")
X_test = pd.read_csv("data/X_test_processed.csv")
y_test = pd.read_csv("data/y_test_processed.csv").values.ravel()

# Step 1: Predict on the test set
print("\nPredicting on the test set...")
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Step 2: Evaluate the model
print("\nEvaluating the model...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Save metrics to a text file
metrics_file = "model_evaluation_metrics.txt"
with open(metrics_file, "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"ROC AUC: {roc_auc:.4f}\n")
print(f"Evaluation metrics saved to {metrics_file}")

# Step 3: Confusion Matrix Visualization
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Fire", "Fire"], yticklabels=["No Fire", "Fire"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
print("Confusion matrix saved as 'confusion_matrix.png'.")

# Step 4: Feature Importance Plot
print("\nPlotting feature importance...")
plt.figure(figsize=(8, 6))
lgb.plot_importance(best_model, max_num_features=10, importance_type="gain")
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Feature importance plot saved as 'feature_importance.png'.")

# Step 5: ROC Curve
print("\nGenerating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.show()
print("ROC curve saved as 'roc_curve.png'.")

print("\nAll visualizations and evaluation results have been saved successfully.")
