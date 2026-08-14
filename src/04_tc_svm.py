import pandas as pd

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
    classification_report
)

# Load the Artificial Training Dataset generated from MATLAB
data = pd.read_csv("Data/ATD.csv")

# Separate the extracted features from the target labels
X = data[["MEAN", "RATIO"]]
y = data["LABEL"]

# Split the dataset into training and testing sets
# The test set is kept unseen during training

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardise the features before training the SVM
# This ensures both features contribute equally

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the TC-SVM model using the parameters reported in Hoang et al. (2021)

svm = SVC(
    kernel="rbf",
    C=1,
    gamma=0.001,
    probability=True,
    random_state=42
)

print("Training TC-SVM...\n")

svm.fit(X_train, y_train)

print("Training completed.\n")

# Predict the class labels for the testing set

y_pred = svm.predict(X_test)

# Calculate prediction probabilities for ROC and AUC

y_prob = svm.predict_proba(X_test)[:, 1]

# Evaluate the model

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