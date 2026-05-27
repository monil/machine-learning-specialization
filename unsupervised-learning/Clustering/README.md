# Clustering

## Overview

Clustering groups similar examples together without using labels. The algorithm discovers the groupings from the data itself.

**Real-world examples:** Grouping similar news articles, market segmentation, social network analysis, image compression.

---

## K-Means Clustering

The most common clustering algorithm. Partitions $m$ examples into $K$ clusters by alternating between assigning examples to the nearest centroid and updating centroids.

### Algorithm Steps

**Step 0 — Random initialization**: Choose $K$ initial centroids randomly from the training examples. Always choose $K$ < $m$ (number of training examples).

**Step 1 — Assign points to cluster centroids**: For each training example $x^{(i)}$, assign it to the closest centroid:
$$c^{(i)} = \arg\min_k \|x^{(i)} - \mu_k\|^2$$

**Step 2 — Move cluster centroids**: Update each centroid to the mean of all examples assigned to it:
$$\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x^{(i)}$$

Repeat Steps 1 and 2 until the assignments stop changing (convergence).

**Edge case**: If a cluster has 0 members, either eliminate it (use $K-1$ clusters) or re-initialize it randomly.

---

## Optimization Objective

K-means minimizes the **distortion cost function** — the average squared distance from each example to its assigned centroid:

$$J(c^{(1)},\ldots,c^{(m)},\mu_1,\ldots,\mu_K) = \frac{1}{m}\sum_{i=1}^{m}\|x^{(i)} - \mu_{c^{(i)}}\|^2$$

- Step 1 (assign) minimizes $J$ over $c^{(i)}$ with $\mu_k$ fixed.
- Step 2 (update) minimizes $J$ over $\mu_k$ with $c^{(i)}$ fixed.
- The cost never increases; K-means is guaranteed to converge.

---

## Random Initialization

K-means can converge to a local optimum depending on initial centroids. To find the global (or a better) optimum:

1. Run K-means **multiple times** (e.g., 50–1000 times) with different random initializations.
2. Compute the cost $J$ for each run.
3. Select the clustering with the **lowest cost**.

This is especially important when $K$ is small (2–10); for large $K$ the first run is usually good enough.

---

## Choosing the Number of Clusters $K$

### Elbow Method
Plot the cost $J$ vs $K$. Look for an "elbow" — a point where adding more clusters yields diminishing returns. This is often ambiguous in practice.

### Downstream Use Case
Often more practical: choose $K$ based on what makes sense for how the clusters will be used (e.g., a clothing company may choose $K=3$ for S/M/L sizing vs $K=5$ for XS/S/M/L/XL).

---

## Practical Notes

- K-means is sensitive to outliers — consider removing them or using a robust variant.
- Always run multiple random initializations and pick the best result.
- Features should be **scaled** before running K-means (Euclidean distance is scale-dependent).
- K-means assumes **spherical, similarly-sized** clusters; it can struggle with elongated or irregular shapes.

---

## Interview Quick-Fire

**Q: What does K-means minimize?**
The distortion cost $J = \frac{1}{m}\sum_i \|x^{(i)} - \mu_{c^{(i)}}\|^2$ — the average squared distance from each point to its assigned centroid.

**Q: Why run K-means multiple times?**
K-means can converge to a local optimum that depends on the initial centroid positions. Multiple random initializations increase the chance of finding the global optimum.

**Q: How do you choose $K$?**
Use the elbow method (plot cost vs $K$, look for a bend) or — more often in practice — choose $K$ based on the downstream task requirements.

**Q: Does K-means require feature scaling?**
Yes. K-means uses Euclidean distance, so features on a larger scale will dominate. Normalize or standardize features before running K-means.
