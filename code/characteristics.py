import pandas as pd
import numpy as np
from collections import defaultdict, Counter

features = pd.read_csv("enron_features.csv")
degrees_df = pd.read_csv("node_degrees.csv")

# Merge on node
df = pd.merge(features, degrees_df, on="node")

# Extract sequences
feature_cols = [col for col in df.columns if col.startswith("deg_")]

df["seq"] = df[feature_cols].apply(lambda row: [x for x in row if x != 0], axis=1)

n = len(df)

# 1. Node heterogeneity (Vn)
df["var"] = df["seq"].apply(lambda s: np.var(s) if len(s) > 1 else 0)
Vn = df["var"].mean()

# 2. Global degree variance
vk = np.var(df["degree"])
Vn_hat = Vn / vk if vk != 0 else 0

# 3. Neighbourhood similarity (S)
seq_tuples = df["seq"].apply(tuple)
counts = Counter(seq_tuples)

S = np.mean([counts[seq] > 1 for seq in seq_tuples])

# 4. Group nodes by degree
groups = defaultdict(list)

for _, row in df.iterrows():
    k = row["degree"]
    groups[k].append(row["seq"])

# 5. Neighbourhood organisation (Ω)
omega_values = []

for k, group in groups.items():
    qp = len(group)
    if qp <= 1:
        continue

    group_tuples = [tuple(seq) for seq in group]
    counts = Counter(group_tuples)

    sigma_p = len(counts)

    numerator = sum(qp - c for c in counts.values())

    omega_p = (sigma_p * numerator) / (qp * qp * (qp - 1))

    omega_values.append(omega_p)

Omega = 1 - np.mean(omega_values) if omega_values else 0

# 6. Hierarchical complexity (R)
R_vals = []

for k, group in groups.items():
    qp = len(group)
    if qp <= 1:
        continue

    max_len = max(len(seq) for seq in group)

    seqs = np.array([
        seq + [0]*(max_len - len(seq))
        for seq in group
    ])

    means = np.mean(seqs, axis=0)

    var_sum = np.sum((seqs - means) ** 2)

    R_k = var_sum / (qp * (qp - 1))

    R_vals.append(R_k)

R = np.mean(R_vals) if R_vals else 0

# 7. Corrected complexity (RΩ)
R_omega_vals = []

for k, group in groups.items():
    qp = len(group)
    if qp <= 1:
        continue

    group_tuples = [tuple(seq) for seq in group]
    counts = Counter(group_tuples)

    sigma_p = len(counts)
    numerator = sum(qp - c for c in counts.values())

    omega_p = (sigma_p * numerator) / (qp * qp * (qp - 1))

    max_len = max(len(seq) for seq in group)

    seqs = np.array([
        seq + [0]*(max_len - len(seq))
        for seq in group
    ])

    means = np.mean(seqs, axis=0)

    var_sum = np.sum((seqs - means) ** 2)

    R_k = omega_p * var_sum / (qp * (qp - 1))

    R_omega_vals.append(R_k)

R_omega = np.mean(R_omega_vals) if R_omega_vals else 0

# Results
print("\n--- Network Characteristics ---")
print(f"Number of nodes: {n}")
print(f"Node heterogeneity (Vn): {Vn:.6f}")
print(f"Normalised heterogeneity (Vn_hat): {Vn_hat:.6f}")
print(f"Neighbourhood similarity (S): {S:.6f}")
print(f"Neighbourhood organisation (Omega): {Omega:.6f}")
print(f"Hierarchical complexity (R): {R:.6f}")
print(f"Corrected complexity (R_Omega): {R_omega:.6f}")