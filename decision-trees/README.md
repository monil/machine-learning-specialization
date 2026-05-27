# Decision Trees

## Overview

A decision tree recursively partitions the feature space by selecting the split that maximizes information gain at each node. The result is a tree of if-else rules that is easy to interpret and requires no feature scaling.

**Use cases:** Tabular/structured data, when interpretability matters, as a building block for ensemble methods (Random Forest, XGBoost).

---

## Key Concepts

**Entropy** — a measure of impurity in a node. A pure node (all one class) has entropy 0; maximum entropy occurs when classes are equally split.

$$H(S) = -\sum_{k} p_k \log_2(p_k)$$

where $p_k$ is the proportion of class $k$ examples in the node.

**Information Gain** — the reduction in entropy achieved by a split. Used to select which feature to split on at each node.

$$IG = H(\text{parent}) - \left[w_L \cdot H(L) + w_R \cdot H(R)\right]$$

where $w_L = \frac{|L|}{|S|}$ and $w_R = \frac{|R|}{|S|}$ are the fraction of examples going left/right.

---

## Decision Tree Building Process

1. Calculate information gain for all features at the current node.
2. Select the feature with the highest information gain.
3. Split the dataset into subsets based on the selected feature.
4. Create left and right branches; route training examples to the appropriate branch.

**Recursive splitting:** Repeat the process on each branch until a stopping criterion is met.

---

## Stopping Criteria

| Criterion | Reason |
|---|---|
| Node reaches 100% single class (entropy = 0) | Node is already pure; no gain possible |
| Tree exceeds maximum depth limit | Prevents overfitting |
| Information gain below threshold | Split would not improve prediction meaningfully |
| Number of examples in node below threshold | Avoids fitting to tiny subsets |

---

## Handling Categorical Features

**One-Hot Encoding** — convert a categorical feature with $k$ categories into $k$ binary columns, each with value 1 for the present category and 0 for all others. This allows decision trees (and most ML algorithms) to handle categorical inputs.

---

## Ensemble Methods

Decision trees have high variance — small changes in training data produce very different trees. Ensembles reduce this:

| Method | How It Works | Key Benefit |
|---|---|---|
| **Random Forest** | Train many trees on bootstrapped data + random feature subsets; average predictions | Reduces variance while keeping low bias |
| **XGBoost** | Gradient boosting: each tree corrects residual errors of previous trees | High accuracy; strong baseline for tabular data |
| **Bagging** | Bootstrap aggregating — train on random samples with replacement and average | Reduces variance |

---

## Practical Notes

- Decision trees are **prone to overfitting** — use max depth or min samples per leaf to regularize.
- They require **no feature scaling** (splits are based on rank order, not magnitude).
- They can handle **mixed feature types** (numerical + categorical) natively.
- For most real-world tabular data, **XGBoost or Random Forest outperforms a single tree**.

---

## Interview Quick-Fire

**Q: What is entropy in the context of decision trees?**
Entropy measures impurity: $H = -\sum_k p_k \log_2 p_k$. A node with all examples from one class has $H=0$ (pure). Maximum entropy occurs when classes are equally distributed.

**Q: What is information gain?**
The reduction in weighted entropy from parent to children after a split: $IG = H(\text{parent}) - [w_L H(L) + w_R H(R)]$. The tree picks the split with the highest IG.

**Q: How does a Random Forest reduce variance?**
By training many trees on different bootstrapped subsets of data and averaging their predictions. Individual trees overfit to their samples, but the average is more stable.

**Q: Why do decision trees not need feature scaling?**
Splits are based on whether a feature value is above or below a threshold, which depends only on the relative ordering of values, not their absolute scale.

**Q: What is one-hot encoding and when do you need it?**
It converts a categorical feature with $k$ values into $k$ binary features. Needed when an algorithm expects numerical input and has no native handling of categorical variables.
