# Recommender Systems

## Overview

Recommender systems predict which items a user is likely to prefer, based on past behavior (ratings, clicks, purchases). They are used in streaming platforms, e-commerce, social media feeds, and ad targeting.

---

## Collaborative Filtering

**Collaborative filtering** learns user and item representations entirely from the observed user–item interactions (ratings). It does not require any features describing the items or users — it "collaborates" across users who have similar taste patterns.

**Intuition**: If user A and user B both liked movies 1, 2, and 3, and user A also liked movie 4, then movie 4 is a good recommendation for user B.

---

## Matrix Factorization

The user–item rating matrix $Y$ (rows = users, columns = items) is typically very sparse. Matrix factorization approximates it as the product of two low-rank matrices:

$$Y \approx X W^T$$

where:
- $X \in \mathbb{R}^{n_u \times k}$: user feature matrix ($n_u$ users, $k$ latent features)
- $W \in \mathbb{R}^{n_m \times k}$: item feature matrix ($n_m$ items, $k$ latent features)
- The predicted rating for user $i$ on item $j$ is: $\hat{y}^{(i,j)} = \mathbf{x}^{(i)} \cdot \mathbf{w}^{(j)} + b^{(j)}$

---

## Cost Function

Minimize the squared error on observed ratings plus regularization:

$$J = \frac{1}{2}\sum_{(i,j):\,r(i,j)=1}\!\bigl(\mathbf{x}^{(i)} \cdot \mathbf{w}^{(j)} + b^{(j)} - y^{(i,j)}\bigr)^2 + \frac{\lambda}{2}\sum_i\|\mathbf{x}^{(i)}\|^2 + \frac{\lambda}{2}\sum_j\|\mathbf{w}^{(j)}\|^2$$

where $r(i,j)=1$ if user $i$ has rated item $j$.

Both $X$ and $W$ are learned simultaneously by gradient descent.

---

## Mean Normalization

Before training, subtract the mean rating of each item from all of its ratings. This ensures the model predicts the mean rating for items with no ratings, rather than zero.

$$y_{\text{norm}}^{(i,j)} = y^{(i,j)} - \mu_j$$

---

## Content-Based vs Collaborative Filtering

| | Content-Based Filtering | Collaborative Filtering |
|---|---|---|
| Requires item features? | Yes (genre, description, etc.) | No |
| Requires user features? | Yes (demographics, preferences) | No |
| Cold start problem | Can recommend new items with features | Cannot recommend new items with no ratings |
| Scales with data | Moderate | Scales well with more ratings |

---

## Practical Notes

- **Cold-start problem**: Collaborative filtering cannot recommend new items that have no ratings yet. Content-based filtering or hybrid approaches are used to address this.
- **Implicit feedback**: When explicit ratings (1–5 stars) are unavailable, clicks/views/purchases can be used as implicit signals.
- **Large-scale systems**: In production, approximate nearest-neighbor search (e.g., retrieval + ranking two-stage architecture) is used to efficiently serve recommendations.

---

## Interview Quick-Fire

**Q: What is collaborative filtering?**
A recommendation technique that predicts a user's preferences based on patterns of similar users, without needing explicit item features. It learns user and item embeddings from the rating matrix.

**Q: What is the cold-start problem?**
New users or new items have no interaction history, so collaborative filtering cannot make good recommendations for them. Content-based filtering or hybrid models are used to handle cold starts.

**Q: What does matrix factorization do?**
It decomposes the sparse user–item rating matrix into two low-rank matrices — user embeddings and item embeddings. Their dot product approximates the original ratings.

**Q: Why use mean normalization in recommender systems?**
It prevents the model from predicting zero (or very low) ratings for items that no user has rated yet. After normalization, the default prediction is the average rating for that item.
