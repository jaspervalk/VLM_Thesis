import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# Dummy data
data = np.random.rand(100, 128)
embedded = TSNE(n_components=2).fit_transform(data)

plt.scatter(embedded[:, 0], embedded[:, 1])
plt.title("t-SNE Visualization Test")
plt.show()


