# Supervised Learning

## Overview

Supervised learning maps inputs $X$ to outputs $Y$ using labeled training examples. The model learns a function $f: X \to Y$ by minimizing error on the training set.

**Real-world applications:**
- Spam filtering
- Speech recognition
- Machine translation
- Online advertising (click prediction)
- Self-driving cars
- Visual inspection / quality control

---

## Problem Types

### 1. Regression
Predicts a **continuous** value.
- Example: housing price prediction — plot features on the x-axis and the target price on the y-axis.

### 2. Classification
Predicts a **discrete class label**.
- Binary: 2 classes (e.g., benign vs malignant).
- Multiclass: more than 2 classes.
- Can have 2 or more input features.

---

## Linear Regression

### Notation

| Symbol | Meaning |
|---|---|
| $x$ (or $x^{(i)}$) | Input feature (for the $i$-th example) |
| $y$ (or $y^{(i)}$) | Output / target variable |
| $m$ | Number of training examples |
| $(x^{(i)}, y^{(i)})$ | The $i$-th training example |
| $w, b$ | Weight and bias parameters |

### Model

**Univariate** (one feature):
$$f_{w,b}(x) = wx + b$$

**Multivariate** ($n$ features):
$$f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$$

### Cost Function

We measure model quality using mean squared error (MSE) and seek parameters that minimize it:

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\bigl(f_{w,b}(x^{(i)}) - y^{(i)}\bigr)^2$$

The $\frac{1}{2}$ factor is conventional — it simplifies the gradient expressions.

**Multivariate version:**
$$J(\mathbf{w},b) = \frac{1}{2m}\sum_{i=1}^{m}\bigl(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\bigr)^2$$

### Gradient Descent

Repeatedly update parameters in the direction of the negative gradient until convergence:

$$w := w - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\bigl(f(x^{(i)}) - y^{(i)}\bigr)\,x^{(i)}$$

$$b := b - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\bigl(f(x^{(i)}) - y^{(i)}\bigr)$$

where $\alpha$ is the **learning rate**.

**Important:** Update $w$ and $b$ **simultaneously** using the same gradients computed before either update.

### Feature Scaling

Features with very different ranges cause gradient descent to converge slowly. Rescale to a similar range using **z-score normalization**:

$$x'_j = \frac{x_j - \mu_j}{\sigma_j}$$

After normalization all features have mean 0 and standard deviation 1.

### Feature Engineering & Polynomial Regression

Extend linear regression to non-linear patterns by creating new features:
$$f = w_1 x + w_2 x^2 + w_3 x^3 + b$$

This is still "linear regression" because it is linear in the parameters $w$; the feature vector just includes polynomial terms.

---

## Logistic Regression

Used for **classification** (not regression despite the name).

### Sigmoid Function

The sigmoid maps any real number to the interval $(0, 1)$, making it suitable for probability outputs:

$$g(z) = \frac{1}{1 + e^{-z}}$$

### Model

$$f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

Interpretation: $f$ outputs $P(y=1 \mid \mathbf{x})$ — the probability that the label is 1.

Predict $\hat{y} = 1$ if $f \geq 0.5$, i.e. when $\mathbf{w} \cdot \mathbf{x} + b \geq 0$.

### Decision Boundary

The decision boundary is the set of points where $f = 0.5$, i.e. where $\mathbf{w} \cdot \mathbf{x} + b = 0$.
- Linear boundary: $w_1 x_1 + w_2 x_2 + b = 0$
- Non-linear boundary: achieved by adding polynomial features before fitting

### Why Not Squared Error for Classification?

Squared error + sigmoid produces a **non-convex** cost function with many local minima. Log-loss produces a convex cost that gradient descent can reliably minimize.

### Cost Function (Log-Loss)

Per-example loss:
$$L\bigl(f(\mathbf{x}^{(i)}), y^{(i)}\bigr) = -y^{(i)}\log\bigl(f(\mathbf{x}^{(i)})\bigr) - (1-y^{(i)})\log\bigl(1-f(\mathbf{x}^{(i)})\bigr)$$

Total cost:
$$J(\mathbf{w},b) = \frac{1}{m}\sum_{i=1}^{m} L\bigl(f(\mathbf{x}^{(i)}), y^{(i)}\bigr)$$

### Gradient Descent for Logistic Regression

The update rules have the **same form** as linear regression:

$$w_j := w_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\bigl(f(\mathbf{x}^{(i)}) - y^{(i)}\bigr)x_j^{(i)}$$

$$b := b - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m}\bigl(f(\mathbf{x}^{(i)}) - y^{(i)}\bigr)$$

The difference is that $f$ is now the sigmoid of the linear combination, not the linear combination itself.

### Regularized Logistic Regression

Add an L2 penalty to the cost to reduce overfitting:

$$J(\mathbf{w},b) = \frac{1}{m}\sum_{i=1}^{m} L(f^{(i)}, y^{(i)}) + \frac{\lambda}{2m}\sum_{j=0}^{n-1}w_j^2$$

The bias term $b$ is not regularized by convention.

---

## Practical Notes

- **Learning rate $\alpha$**: If cost oscillates or increases → $\alpha$ too large. If cost decreases very slowly → $\alpha$ too small. Try values like 0.001, 0.01, 0.1, 1.
- **Multivariate regression**: Use `np.dot(w, x)` (vectorized) rather than a for-loop — faster and cleaner.
- **Feature scaling**: Always scale before running gradient descent on multivariate problems.
- **Extend to multivariate**: Replace scalar $w$ with weight vector $\mathbf{w}$; extend gradients accordingly.
- **Regularization** (Ridge/L2, Lasso/L1): Helps reduce overfitting. See [Overfitting notes](Overfitting/OverFitting.md).
- **For classification**: Consider logistic regression, decision trees, SVMs, and neural networks.

---

## Interview Quick-Fire

**Q: What is the difference between regression and classification?**
Regression predicts a continuous output (e.g., house price); classification predicts a discrete class label (e.g., spam / not spam).

**Q: Why do we use the $\frac{1}{2}$ in MSE?**
It is a convention that simplifies the gradient — the 2 from the power cancels the $\frac{1}{2}$, giving cleaner update rules.

**Q: Why can't we use MSE as the cost function for logistic regression?**
With logistic regression, MSE produces a non-convex cost surface with many local minima, making gradient descent unreliable. Log-loss (cross-entropy) gives a convex cost.

**Q: What does the learning rate control?**
The size of each parameter update step. Too large: gradient descent diverges or oscillates. Too small: convergence is very slow.

**Q: What is feature scaling and why does it matter?**
Normalizing features to a similar scale prevents gradient descent from taking inefficient zig-zag paths through the cost landscape, leading to much faster convergence.
