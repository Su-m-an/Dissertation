import joblib

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Load the Artificial Training Dataset

data = pd.read_csv("Data/ATD.csv")

# Separate the extracted features and labels

X = data[["MEAN", "RATIO"]]
y = data["LABEL"]

# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardise the feature values
# Although Random Forest does not require scaling,
# we keep the preprocessing consistent across all models.

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the Random Forest classifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest...\n")

rf.fit(X_train, y_train)

print("Training completed.\n")

# Predict the testing samples

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:,1]

# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC Score: {auc:.4f}")

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# Save evaluation metrics

metrics = pd.DataFrame({
    "Metric":[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ],
    "Value":[
        accuracy,
        precision,
        recall,
        f1,
        auc
    ]
})

metrics.to_csv(
    "results/random_forest_metrics.csv",
    index=False
)

# Save the trained model and scaler for reuse without retraining

joblib.dump(rf, "saved_models/random_forest.joblib")
joblib.dump(scaler, "saved_models/random_forest_scaler.joblib")

# Calculate feature importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

importance.to_csv(
    "results/random_forest_feature_importance.csv",
    index=False
)

# Plot feature importance

plt.figure(figsize=(6,4))

plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig(
    "figures/random_forest_feature_importance.png",
    dpi=300
)

plt.close()

print("\nFeature importance saved successfully.")