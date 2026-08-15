import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

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

# Standardise the feature values before training

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the TC-SVM classifier

svm = SVC(
    kernel="rbf",
    C=1,
    gamma=0.001,
    probability=True,
    random_state=42
)

print("Training TC-SVM...")

svm.fit(X_train, y_train)

print("Model trained successfully.")

# Calculate prediction probabilities

y_prob = svm.predict_proba(X_test)[:,1]

# Calculate ROC values

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

auc = roc_auc_score(y_test, y_prob)

print(f"AUC Score : {auc:.4f}")

# Plot the ROC Curve

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"TC-SVM (AUC = {auc:.4f})"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--",
    linewidth=2
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for TC-SVM")
plt.legend(loc="lower right")

plt.grid(True)

# Saved under a distinct name from 06_model_evaluation.py, which also
# produces a TC-SVM ROC curve (figures/roc_curve_tc_svm.png). Without this,
# the two scripts would silently overwrite each other's figure depending on run order.
plt.savefig("figures/roc_curve_tc_svm_standalone.png", dpi=300)

plt.close()

print("ROC curve saved inside the figures folder.")