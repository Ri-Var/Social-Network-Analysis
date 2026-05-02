import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv("enron_features.csv")

print("Shape:", df.shape)

pca = PCA(n_components=2)
reduced = pca.fit_transform(df)

plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], s=5, alpha=0.6)

plt.title("Node Visualization (PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()