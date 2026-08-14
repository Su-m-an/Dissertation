import pandas as pd

# Load the evaluation metrics for each trained model

svm = pd.read_csv("results/tc_svm_metrics.csv")
rf = pd.read_csv("results/random_forest_metrics.csv")
xgb = pd.read_csv("results/xgboost_metrics.csv")

# Convert each metrics file into a dictionary for easier access

svm_metrics = dict(zip(svm["Metric"], svm["Value"]))
rf_metrics = dict(zip(rf["Metric"], rf["Value"]))
xgb_metrics = dict(zip(xgb["Metric"], xgb["Value"]))

# Create a comparison table

comparison = pd.DataFrame({

    "Model": [
        "TC-SVM",
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        svm_metrics["Accuracy"],
        rf_metrics["Accuracy"],
        xgb_metrics["Accuracy"]
    ],

    "Precision": [
        svm_metrics["Precision"],
        rf_metrics["Precision"],
        xgb_metrics["Precision"]
    ],

    "Recall": [
        svm_metrics["Recall"],
        rf_metrics["Recall"],
        xgb_metrics["Recall"]
    ],

    "F1 Score": [
        svm_metrics["F1 Score"],
        rf_metrics["F1 Score"],
        xgb_metrics["F1 Score"]
    ],

    "AUC": [
        svm_metrics["AUC"],
        rf_metrics["AUC"],
        xgb_metrics["AUC"]
    ]

})

# Round the values for easier reading

comparison = comparison.round(4)

# Save the comparison table

comparison.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\nModel Comparison\n")

print(comparison)

print("\nComparison table saved successfully.")