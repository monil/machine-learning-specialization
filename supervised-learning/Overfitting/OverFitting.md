# Overfitting & Regularization

## Key Concepts

| Term | Description |
|---|---|
| **Underfitting** (high bias) | Model is too simple; does not fit the training set well. Both $J_{train}$ and $J_{cv}$ are high. |
| **Generalization** | Model fits the training set pretty well and also performs well on unseen data. |
| **Overfitting** (high variance) | Model fits the training set extremely well (including noise) but fails on new data. $J_{train}$ is low; $J_{cv}$ is much higher. |

---

## Addressing Overfitting

1. **Collect more training examples** — gives the model more signal, less chance of memorizing noise.
2. **Select features to include/exclude** — removing irrelevant features simplifies the model.
3. **Regularization** — reduce the magnitude of parameters ($w_j$) so the model is smoother and generalizes better. Does not eliminate features entirely, just shrinks their influence.

---

## Regularization

### L2 Regularization (Ridge)

Adds the sum of squared weights to the cost function:

**Regularized linear regression cost:**
$$J(\mathbf{w},b) = \frac{1}{2m}\sum_{i=0}^{m-1}\bigl(f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}\bigr)^2 + \frac{\lambda}{2m}\sum_{j=0}^{n-1}w_j^2$$

**Regularized logistic regression cost:**
$$J(\mathbf{w},b) = \frac{1}{m}\sum_{i=0}^{m-1}\Bigl[-y^{(i)}\log\bigl(f(\mathbf{x}^{(i)})\bigr) - (1-y^{(i)})\log\bigl(1-f(\mathbf{x}^{(i)})\bigr)\Bigr] + \frac{\lambda}{2m}\sum_{j=0}^{n-1}w_j^2$$

### L1 Regularization (Lasso)

Adds the sum of absolute weights:
$$\text{penalty} = \frac{\lambda}{2m}\sum_{j=0}^{n-1}|w_j|$$

- L1 drives some weights to **exactly zero**, producing sparse models (automatic feature selection).
- L2 shrinks all weights toward zero but rarely to exactly zero.

### Regularization Parameter $\lambda$

| $\lambda$ | Effect |
|---|---|
| $\lambda = 0$ | No regularization; model may overfit |
| $\lambda$ too large | All weights shrink to near zero; model underfits |
| $\lambda$ just right | Balance between fitting the training data and keeping weights small |

---

## Gradient Descent with Regularization

The update rule for $w_j$ gains an extra shrinkage term:

$$w_j := w_j - \alpha\left[\frac{1}{m}\sum_{i=0}^{m-1}\bigl(f(\mathbf{x}^{(i)}) - y^{(i)}\bigr)x_j^{(i)} + \frac{\lambda}{m}w_j\right]$$

This is equivalent to multiplying $w_j$ by $(1 - \alpha\frac{\lambda}{m})$ before adding the gradient — the weight is "decayed" slightly on every step.

The bias $b$ is **not** regularized by convention.

---

## Bias–Variance Tradeoff

| Symptom | Diagnosis | Fix |
|---|---|---|
| $J_{train}$ high, $J_{cv}$ high | High bias (underfitting) | Add features, increase model complexity, decrease $\lambda$ |
| $J_{train}$ low, $J_{cv} \gg J_{train}$ | High variance (overfitting) | More training data, fewer features, increase $\lambda$ |
| $J_{train}$ low, $J_{cv}$ low | Good fit | — |

---

## Interview Quick-Fire

**Q: What is the difference between bias and variance?**
Bias is error from wrong assumptions (model too simple). Variance is error from sensitivity to small fluctuations in training data (model too complex). High bias → underfitting; high variance → overfitting.

**Q: What does regularization do to the weights?**
It penalizes large weights, pushing them toward zero. This reduces model complexity and helps the model generalize better.

**Q: When would you prefer L1 over L2 regularization?**
When you want automatic feature selection — L1 can drive irrelevant feature weights exactly to zero, producing a sparse model. L2 only shrinks weights.

**Q: What happens if $\lambda$ is too large?**
The model underfits — all weights are penalized so heavily that they approach zero, and the model loses the ability to capture patterns in the data.
