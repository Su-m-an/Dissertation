import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset generated from the MATLAB simulation
data = pd.read_csv("Data/ATD.csv")

# Separate attack and non-attack samples
attack = data[data["LABEL"] == 1]
non_attack = data[data["LABEL"] == 0]

# Plot the relationship between the two extracted features

plt.figure(figsize=(8, 6))

plt.scatter(
    non_attack["MEAN"],
    non_attack["RATIO"],
    s=5,
    alpha=0.4,
    label="Non-Attack"
)

plt.scatter(
    attack["MEAN"],
    attack["RATIO"],
    s=5,
    alpha=0.4,
    label="Attack"
)

plt.xlabel("MEAN")
plt.ylabel("RATIO")
plt.title("Scatter Plot of MEAN vs RATIO")
plt.legend()

# Save the figure for the dissertation
plt.savefig("figures/scatter_mean_ratio.png", dpi=300)

plt.close()

# Compare the distribution of the MEAN feature

plt.figure(figsize=(8, 6))

plt.hist(
    non_attack["MEAN"],
    bins=50,
    alpha=0.6,
    label="Non-Attack"
)

plt.hist(
    attack["MEAN"],
    bins=50,
    alpha=0.6,
    label="Attack"
)

plt.xlabel("MEAN")
plt.ylabel("Frequency")
plt.title("Distribution of the MEAN Feature")
plt.legend()

plt.savefig("figures/mean_distribution.png", dpi=300)

plt.close()

# Compare the distribution of the RATIO feature

plt.figure(figsize=(8, 6))

plt.hist(
    non_attack["RATIO"],
    bins=50,
    alpha=0.6,
    label="Non-Attack"
)

plt.hist(
    attack["RATIO"],
    bins=50,
    alpha=0.6,
    label="Attack"
)

plt.xlabel("RATIO")
plt.ylabel("Frequency")
plt.title("Distribution of the RATIO Feature")
plt.legend()

plt.savefig("figures/ratio_distribution.png", dpi=300)

plt.close()

print("\nVisualisation completed successfully.")
print("Figures have been saved in the figures folder.")