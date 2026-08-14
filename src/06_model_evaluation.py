import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

# Load the Artificial Training Dataset

data = pd.read_csv("Data/ATD.csv")

# Separate features and labels

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

# Standardise the features

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the TC-SVM model

svm = SVC(
    kernel="rbf",
    C=1,
    gamma=0.001,
    probability=True,
    random_state=42
)

print("Training TC-SVM...\n")

svm.fit(X_train, y_train)

print("Model trained successfully.\n")

# Generate predictions

y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# Save metrics

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
    "results/tc_svm_metrics.csv",
    index=False
)

# Save confusion matrix

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Non-Attack","Actual Attack"],
    columns=["Predicted Non-Attack","Predicted Attack"]
)

cm_df.to_csv(
    "results/tc_svm_confusion_matrix.csv"
)

# Save classification report

report = classification_report(y_test, y_pred)

with open(
    "results/tc_svm_classification_report.txt",
    "w"
) as file:
    file.write(report)

# Plot confusion matrix

plt.figure(figsize=(6,5))

plt.imshow(cm, cmap="Blues")

plt.colorbar()

plt.xticks([0,1],["Non","Attack"])
plt.yticks([0,1],["Non","Attack"])

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i,j],
            ha="center",
            va="center",
            fontsize=12
        )

plt.tight_layout()

plt.savefig(
    "figures/confusion_matrix_tc_svm.png",
    dpi=300
)

# Plot ROC Curve

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.4f}"
)

plt.plot(
    [0,1],
    [0,1],
    "--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()

plt.grid(True)

plt.savefig(
    "figures/roc_curve_tc_svm.png",
    dpi=300
)

# Plot Precision-Recall Curve

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

plt.figure(figsize=(7,6))

plt.plot(
    recall_curve,
    precision_curve,
    linewidth=2
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")

plt.grid(True)

plt.savefig(
    "figures/precision_recall_tc_svm.png",
    dpi=300
)

plt.show()

print("="*50)
print("Evaluation completed successfully.")
print("="*50)

print(metrics)

print("\nAll evaluation files saved.")