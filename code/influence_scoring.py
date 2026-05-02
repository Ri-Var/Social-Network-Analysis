import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
features = pd.read_csv("enron_features_full.csv")
pca = pd.read_csv("enron_pca_full.csv")
degree_df = pd.read_csv("node_degrees.csv")

# Merge on node
df = features.merge(pca, on="node")

# EXTRACT DEGREE
df = df.merge(degree_df, on="node")

# NORMALIZATION
df["degree_log"] = np.log1p(df["degree"])

scaler = StandardScaler()
df[["degree_norm", "PC1_norm"]] = scaler.fit_transform(
    df[["degree_log", "PC1"]]
)

# INFLUENCE SCORE
alpha = 0.4  # weight

df["influence_score"] = alpha * df["degree_norm"] + (1 - alpha) * df["PC1_norm"]

# RANK NODES
df_sorted = df.sort_values("influence_score", ascending=False)
df_sorted = df_sorted[["node", "PC1", "degree", "influence_score"]]

# Top influencers
top_k = 50
top_influencers = df_sorted.head(top_k)


#print("\nTop Influencers:")
#print(top_influencers[["node", "influence_score"]].head(10))

top_influencers.to_csv("top_influencers.csv", index=False)
df_sorted.to_csv("all_nodes_ranked.csv", index=False)
