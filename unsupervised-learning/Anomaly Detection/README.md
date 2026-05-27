# Anomaly Detection

## Overview

Anomaly detection identifies unusual examples by modeling the probability distribution of normal data. An example is flagged as an anomaly if its probability under the learned distribution falls below a threshold $\varepsilon$.

**Real-world examples:** Fraud detection, manufacturing quality control, network intrusion detection, aircraft engine monitoring.

---

## Key Concepts

**Gaussian (Normal) Distribution**: Used to model each feature's distribution.

$$p(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\,\sigma}\,e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**Density Estimation**: Model the joint probability of all features assuming they are independent:

$$p(\mathbf{x}) = \prod_{j=1}^{n} p(x_j;\mu_j,\sigma_j^2)$$

---

## Algorithm Steps

1. **Choose features**: Select $n$ features $x_1, \ldots, x_n$ that are likely indicative of anomalous behavior.

2. **Fit parameters**: Estimate the mean and variance for each feature using the training set:
   $$\mu_j = \frac{1}{m}\sum_{i=1}^{m}x_j^{(i)}, \qquad \sigma_j^2 = \frac{1}{m}\sum_{i=1}^{m}\bigl(x_j^{(i)} - \mu_j\bigr)^2$$

3. **Compute probability**: For a new example $\mathbf{x}$, compute $p(\mathbf{x}) = \prod_{j=1}^{n} p(x_j;\mu_j,\sigma_j^2)$.

4. **Flag anomalies**: Predict anomaly if $p(\mathbf{x}) < \varepsilon$.

---

## Evaluation

Because anomaly detection has very few positive (anomalous) examples, you need a small labeled validation set to tune $\varepsilon$:

1. Fit $p(\mathbf{x})$ on the unlabeled training set (mostly normal examples).
2. On the cross-validation set, predict $\hat{y} = 1$ (anomaly) if $p(\mathbf{x}) < \varepsilon$, else $\hat{y} = 0$.
3. Tune $\varepsilon$ using F1 score or precision/recall (not accuracy — class imbalance makes accuracy misleading).

---

## Anomaly Detection vs Supervised Learning

| | Anomaly Detection | Supervised Learning |
|---|---|---|
| Positive examples | Very few (< ~20) | Many |
| Negative examples | Large number | Large number |
| Future anomalies | May look very different from past | Similar to training distribution |
| Example use case | Fraud, defect detection | Spam classification |

**Rule of thumb**: If you have very few positive (anomalous) examples, anomaly detection is preferred. With many labeled positive examples, use supervised learning.

---

## Practical Notes

- **Feature engineering matters**: Choose features that take unusually large or small values for anomalous examples.
- **Non-Gaussian features**: Apply transformations ($\log(x)$, $\sqrt{x}$, etc.) to make features more Gaussian before fitting.
- **Multivariate Gaussian**: If features are correlated, the product-of-univariates assumption may miss anomalies that only appear in feature combinations. Use a full multivariate Gaussian if $m \gg n$.

---

## Interview Quick-Fire

**Q: How does anomaly detection work?**
It fits a probability distribution (usually Gaussian) to normal training data. A new example with probability $p(\mathbf{x}) < \varepsilon$ is flagged as an anomaly.

**Q: When should you use anomaly detection vs supervised classification?**
Use anomaly detection when you have very few labeled anomalous examples (it learns only from normal data). Use supervised classification when you have enough labeled examples of both classes.

**Q: Why is accuracy a bad metric for anomaly detection?**
Anomalies are rare — a model that always predicts "normal" achieves near-100% accuracy but is useless. Use precision, recall, or F1 score instead.

**Q: What if features are not Gaussian?**
Apply transformations like $\log(x+c)$ or $x^{1/2}$ to make them more Gaussian before fitting the density model.
