import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm
import random
from scipy.stats import wilcoxon

# SETTINGS
FILE_PATH = "Email-Enron.txt"
TOP_K = 20
NUM_REALIZATIONS = 10
SEED = 42

rng = random.Random(SEED)

# 1. LOAD GRAPH
def load_graph(file_path):
    G = nx.Graph()
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "FromNodeId" in line:
                continue
            u, v = line.split()
            if u != v:
                G.add_edge(u, v)
    return G

# 2. NEIGHBOURHOOD DEGREE SEQUENCES (TOP_K truncated)
def compute_sequences(G, nodes, TOP_K=20):
    sequences = {}
    for node in nodes:
        degrees = sorted(
            [G.degree(n) for n in G.neighbors(node)],
            reverse=True
        )

        top_degrees = degrees[:TOP_K]

        if len(top_degrees) < TOP_K:
            top_degrees += [0] * (TOP_K - len(top_degrees))

        sequences[node] = top_degrees

    return sequences

# 3. METRICS
def compute_Vn(sequences):
    vars_ = []
    for seq in sequences.values():
        if len(seq) > 1:
            vars_.append(np.var(seq))
    return np.mean(vars_) if vars_ else 0

def compute_S(sequences):
    seq_tuples = [tuple(v) for v in sequences.values()]
    counts = Counter(seq_tuples)
    return np.mean([counts[s] > 1 for s in seq_tuples])

def group_by_degree(G, sequences):
    groups = defaultdict(list)
    for node, seq in sequences.items():
        k = G.degree(node)
        groups[k].append(seq)
    return groups

def compute_Omega(groups):
    omega_vals = []
    for k, group in groups.items():
        qp = len(group)
        if qp <= 1:
            continue

        counts = Counter(tuple(s) for s in group)
        sigma_p = len(counts)

        numerator = sum(qp - c for c in counts.values())
        omega_p = (sigma_p * numerator) / (qp * qp * (qp - 1))

        omega_vals.append(omega_p)

    return 1 - np.mean(omega_vals) if omega_vals else 0

def compute_R(groups):
    R_vals = []
    for k, group in groups.items():
        qp = len(group)
        if qp <= 1:
            continue

        seqs = np.array(group)
        means = np.mean(seqs, axis=0)

        var_sum = np.sum((seqs - means) ** 2)
        R_vals.append(var_sum / (qp * (qp - 1)))

    return np.mean(R_vals) if R_vals else 0

def compute_R_omega(groups):
    R_vals = []
    for k, group in groups.items():
        qp = len(group)
        if qp <= 1:
            continue

        counts = Counter(tuple(s) for s in group)
        sigma_p = len(counts)

        numerator = sum(qp - c for c in counts.values())
        omega_p = (sigma_p * numerator) / (qp * qp * (qp - 1))

        seqs = np.array(group)
        means = np.mean(seqs, axis=0)

        var_sum = np.sum((seqs - means) ** 2)

        R_vals.append(omega_p * var_sum / (qp * (qp - 1)))

    return np.mean(R_vals) if R_vals else 0

# 4. METRIC WRAPPER
def compute_all_metrics(G, nodes, TOP_K=20, export=False, filename="sequences.csv"):
    sequences = compute_sequences(G, nodes, TOP_K)

    if export:
        data = [[node] + seq for node, seq in sequences.items()]
        cols = ["node"] + [f"deg_{i+1}" for i in range(TOP_K)]
        pd.DataFrame(data, columns=cols).to_csv(filename, index=False)
        print(f"Saved → {filename}")

    Vn = compute_Vn(sequences)
    S = compute_S(sequences)

    groups = group_by_degree(G, sequences)

    Omega = compute_Omega(groups)
    R = compute_R(groups)
    R_omega = compute_R_omega(groups)

    return {
        "Vn": Vn,
        "S": S,
        "Omega": Omega,
        "R": R,
        "R_omega": R_omega
    }

# 5. CONFIGURATION MODEL (FULL GRAPH)
def generate_config_model(G):
    degree_sequence = [d for _, d in G.degree()]
    CM = nx.configuration_model(degree_sequence)

    CM = nx.Graph(CM)
    CM.remove_edges_from(nx.selfloop_edges(CM))

    return CM

# 6. MAIN
G = load_graph(FILE_PATH)
nodes = sorted(G.nodes())

print("\n--- REAL NETWORK ---")
real_metrics = compute_all_metrics(
    G,
    nodes,
    TOP_K=TOP_K,
    export=True,
    filename="enron_features_full.csv"
)
print(real_metrics)

# 7. CONFIGURATION MODELS
print("\n--- CONFIGURATION MODELS ---")
cm_results = []

for i in range(NUM_REALIZATIONS):
    print(f"Run {i+1}/{NUM_REALIZATIONS}")

    CM = generate_config_model(G)

    cm_nodes = list(CM.nodes())

    metrics = compute_all_metrics(
        CM,
        cm_nodes,
        TOP_K=TOP_K
    )

    cm_results.append(metrics)

cm_df = pd.DataFrame(cm_results)

# 8. COMPARISON
print("\n--- COMPARISON ---")

for metric in real_metrics.keys():
    real_val = real_metrics[metric]
    cm_vals = cm_df[metric]

    stat, p = wilcoxon(cm_vals - real_val)

    print(f"\n{metric}:")
    print(f"Real: {real_val:.5f}")
    print(f"CM mean: {cm_vals.mean():.5f}")
    print(f"p-value: {p:.5e}")