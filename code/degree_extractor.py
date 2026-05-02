import networkx as nx
import pandas as pd

FILE_PATH = "Email-Enron.txt"

# 1. LOAD GRAPH
G = nx.Graph()

with open(FILE_PATH, "r") as f:
    for line in f:
        line = line.strip()

        # Skip comments / headers
        if not line or line.startswith("#") or "FromNodeId" in line:
            continue

        u, v = line.split()

        # Remove self-loops
        if u != v:
            G.add_edge(u, v)

print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 2. COMPUTE DEGREES
data = [(node, deg) for node, deg in G.degree()]

# Convert to DataFrame
df = pd.DataFrame(data, columns=["node", "degree"])

# Optional: sort by degree (descending)
df = df.sort_values(by="degree", ascending=False)

# 3. SAVE CSV
df.to_csv("node_degrees.csv", index=False)

print("Saved → node_degrees.csv")