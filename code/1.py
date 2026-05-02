# This CSV captures the questions, what does a local network around a node look it? How dense is it?

# We take a local subgraph at a cutoff distance of 2, and sort the degrees of the nodes in that local subgraph and take the TOP_K largest values. Hence, for each node we have 20 features and they collectively describe how dense, the network around a node is. 

import networkx as nx
import random
import pandas as pd
from tqdm import tqdm  

FILE_PATH = "Email-Enron.txt"
N = 20000      
K = 1           
TOP_K = 20      

G = nx.Graph()

with open(FILE_PATH, "r") as f:
    for line in f:
        line = line.strip()

        if not line or line.startswith("#") or "FromNodeId" in line:
            continue

        u, v = line.split()
        if u != v:
            G.add_edge(u, v)

print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

SEED = 42

nodes = list(G.nodes())
result = [[node, deg] for node, deg in G.degree()]
sample_t = random.Random(SEED).sample(result, min(N, len(nodes)))

# Node-Degree CSV
df = pd.DataFrame(sample_t, columns=["node", "degree"])
df.to_csv("node_degrees.csv", index=False)
print("Saved node_degrees.csv")

# Neighborhood Degree Sequences CSV (paper-correct)
data = []
for node, _ in tqdm(sample_t):
    neighbors = list(G.neighbors(node))  
    degrees = [G.degree(n) for n in neighbors]
    degrees.sort()
    top_degrees = degrees[-TOP_K:] if len(degrees) >= TOP_K else degrees

    if len(top_degrees) < TOP_K:
        top_degrees = [0] * (TOP_K - len(top_degrees)) + top_degrees

    data.append([node] + top_degrees)

columns = ["node"] + [f"deg_{i+1}" for i in range(TOP_K)]
df = pd.DataFrame(data, columns=columns)

df.to_csv("enron_features.csv", index=False)
print("saved enron_features.csv")