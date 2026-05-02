import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
df = pd.read_csv("enron_features_full.csv")

node_ids = df["node"]
feature_cols = [col for col in df.columns if col.startswith("deg_")]

X = df[feature_cols].values

# 2. PREPROCESSING (CRITICAL)

# Log transform (handles heavy tail)
X_log = np.log1p(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)

# 3. PCA (ALL 20 COMPONENTS)
pca = PCA(n_components=len(feature_cols))
X_pca = pca.fit_transform(X_scaled)

# 4. SAVE PCA OUTPUT
df_pca = pd.DataFrame(
    X_pca,
    columns=[f"PC{i+1}" for i in range(len(feature_cols))]
)

df_pca.insert(0, "node", node_ids)
df_pca.to_csv("enron_pca_full.csv", index=False)

# 5. VARIANCE EXPLAINED
variance_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(feature_cols))],
    "Variance_Explained": pca.explained_variance_ratio_,
    "Cumulative_Variance": np.cumsum(pca.explained_variance_ratio_)
})

variance_df.to_csv("pca_variance_explained.csv", index=False)

print("\n--- Variance Explained ---")
print(variance_df.head(10))

# 6. COMPONENT LOADINGS (HOW PCs ARE FORMED)
loadings = pd.DataFrame(
    pca.components_,
    columns=feature_cols,
    index=[f"PC{i+1}" for i in range(len(feature_cols))]
)

loadings.to_csv("pca_loadings.csv")

print("\n--- PCA Loadings (first few PCs) ---")
print(loadings.head())

# 7. FEATURE CONTRIBUTION (%)
contributions = (pca.components_ ** 2)
contributions_percent = contributions * 100

contrib_df = pd.DataFrame(
    contributions_percent,
    columns=feature_cols,
    index=[f"PC{i+1}" for i in range(len(feature_cols))]
)

contrib_df.to_csv("pca_feature_contributions.csv")

print("\n--- Feature Contribution to PC1 ---")
print(contrib_df.loc["PC1"])