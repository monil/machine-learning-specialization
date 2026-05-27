# Unsupervised Learning

## Overview

Unsupervised learning works with data that has only input features $\mathbf{x}$ and no output label $y$. The algorithm must find structure, patterns, or compact representations entirely on its own.

**When to use unsupervised learning**: When labeled data is unavailable, expensive, or you want to explore the data's natural structure before labeling.

---

## Types of Unsupervised Learning

### 1. Clustering
Groups similar examples together into clusters.
- No predefined labels; the algorithm discovers the groupings.
- Examples: customer segmentation, news article grouping, gene expression analysis.
- Key algorithm: **K-Means** — see [Clustering/README.md](Clustering/README.md).

### 2. Anomaly Detection
Identifies examples that are unusual or do not fit the learned distribution.
- Models a probability distribution over normal data; flags low-probability examples.
- Examples: fraud detection, manufacturing defect detection, network intrusion detection.
- Key approach: **Gaussian density estimation** — see [Anomaly Detection/README.md](Anomaly%20Detection/README.md).

### 3. Dimensionality Reduction
Compresses high-dimensional data into fewer dimensions while preserving as much structure as possible.
- Reduces storage and computation; can remove noise; enables visualization.
- Examples: PCA (Principal Component Analysis), t-SNE (visualization).
- Use case: compress 1000-feature data to 50 features before feeding to another model.

---

## Unsupervised vs Supervised Learning

| | Supervised | Unsupervised |
|---|---|---|
| Labels | Required ($x$, $y$ pairs) | Not required (only $x$) |
| Goal | Predict $y$ for new $x$ | Discover structure in $x$ |
| Examples | Spam detection, price prediction | Clustering, anomaly detection |

---

## Interview Quick-Fire

**Q: What is the key difference between supervised and unsupervised learning?**
Supervised learning requires labeled examples $(x, y)$ to learn a mapping from inputs to outputs. Unsupervised learning has only inputs $x$ and must discover structure (groups, anomalies, compressed representations) without ground-truth labels.

**Q: Give an example where unsupervised learning is preferred over supervised.**
Fraud detection early in a product's life — there are very few labeled fraud examples, so you cannot train a supervised classifier. Anomaly detection models the distribution of normal behavior without needing fraud labels.
