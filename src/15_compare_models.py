import pandas as pd
import matplotlib.pyplot as plt

models = {
    "TC-SVM": "results/tc_svm_metrics.csv",
    "Random Forest": "results/random_forest_metrics.csv",
    "XGBoost": "results/xgboost_metrics.csv",
    "MLP": "results/mlp_metrics.csv",
    "Autoencoder": "results/autoencoder_metrics.csv",
    "LSTM": "results/lstm_metrics.csv"
}

rows = []

for model, file in models.items():

    df = pd.read_csv(file)

    metrics = dict(zip(df["Metric"], df["Value"]))

    rows.append({

        "Model": model,
        "Accuracy": metrics.get("Accuracy"),
        "Precision": metrics.get("Precision"),
        "Recall": metrics.get("Recall"),
        "F1 Score": metrics.get("F1 Score"),
        "AUC": metrics.get("AUC", metrics.get("ROC AUC")),
        "Training Time (s)": metrics.get("Training Time (s)", None)

    })

comparison = pd.DataFrame(rows)

comparison = comparison.sort_values(
    by="Accuracy",
    ascending=False
)

comparison = comparison.round(4)

comparison.to_csv(
    "results/final_model_comparison.csv",
    index=False
)

comparison.to_excel(
    "results/final_model_comparison.xlsx",
    index=False
)

best = comparison.iloc[0]

with open("results/best_model.txt","w") as f:

    f.write(f"Best Model : {best['Model']}\n")
    f.write(f"Accuracy : {best['Accuracy']:.4f}\n")
    f.write(f"Precision : {best['Precision']:.4f}\n")
    f.write(f"Recall : {best['Recall']:.4f}\n")
    f.write(f"F1 Score : {best['F1 Score']:.4f}\n")
    f.write(f"AUC : {best['AUC']:.4f}\n")

print("\nFinal Comparison\n")

print(comparison)

plt.figure(figsize=(10,6))

plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.ylabel("Accuracy")

plt.xlabel("Model")

plt.title("Model Accuracy Comparison")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "figures/model_accuracy_comparison.png",
    dpi=300
)

plt.close()

print("\nComparison table saved.")
print("Accuracy figure saved.")
print("Best model saved.")