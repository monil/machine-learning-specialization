# ML Specialization — Concept Glossary

Quick-reference definitions for all major topics covered in Andrew Ng's Machine Learning Specialization. Use this alongside the topic READMEs for deeper explanations.

## Topic Index

| Topic | Notes File | Key Concepts |
|---|---|---|
| Supervised Learning | [supervised-learning/README.md](supervised-learning/README.md) | Linear regression, logistic regression, cost functions, gradient descent |
| Overfitting & Regularization | [supervised-learning/Overfitting/OverFitting.md](supervised-learning/Overfitting/OverFitting.md) | Bias/variance, L1/L2 regularization |
| Neural Networks | [neural-networks/README.md](neural-networks/README.md) | Layers, activations, backprop, Adam, softmax |
| Decision Trees | [decision-trees/README.md](decision-trees/README.md) | Entropy, information gain, splitting |
| Unsupervised Learning | [unsupervised-learning/README.md](unsupervised-learning/README.md) | Clustering, anomaly detection, dimensionality reduction |
| K-Means Clustering | [unsupervised-learning/Clustering/README.md](unsupervised-learning/Clustering/README.md) | Centroids, cost function, elbow method |
| Anomaly Detection | [unsupervised-learning/Anomaly Detection/README.md](unsupervised-learning/Anomaly%20Detection/README.md) | Gaussian density, threshold ε |
| Recommender Systems | [unsupervised-learning/Recommender Systems/README.md](unsupervised-learning/Recommender%20Systems/README.md) | Collaborative filtering, matrix factorization |

---

## Linear Regression

| Term | Definition |
|---|---|
| Feature ($x$) | Input variable used to make a prediction |
| Target ($y$) | Output variable we are trying to predict |
| Training example | A single $(x^{(i)}, y^{(i)})$ pair |
| $m$ | Number of training examples |
| Model | $f_{w,b}(x) = wx + b$ (univariate); $f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b$ (multivariate) |
| Cost function (MSE) | $J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}(f(x^{(i)}) - y^{(i)})^2$ |
| Gradient descent | Iterative update: $w := w - \alpha\frac{\partial J}{\partial w}$, $b := b - \alpha\frac{\partial J}{\partial b}$ |
| Learning rate ($\alpha$) | Step size for each gradient descent update; too large → diverge, too small → slow |
| Feature scaling | Normalize features (e.g. z-score) so gradient descent converges faster |
| z-score normalization | $x' = \frac{x - \mu}{\sigma}$; result has mean 0, std 1 |
| Polynomial regression | Extend linear regression by adding $x^2, x^3, \ldots$ as features |
| Vectorization | Use `np.dot(w, x)` instead of for-loops; exploits hardware parallelism |

---

## Logistic Regression

| Term | Definition |
|---|---|
| Sigmoid function | $g(z) = \frac{1}{1+e^{-z}}$; maps any real number to $(0,1)$ |
| Logistic model | $f_{\mathbf{w},b}(\mathbf{x}) = g(\mathbf{w} \cdot \mathbf{x} + b)$; output = P(y=1 \| x) |
| Decision boundary | Threshold at $g(z) = 0.5$, i.e. $z = 0$; can be linear or non-linear |
| Why not MSE? | Squared error + sigmoid produces a non-convex cost (many local minima); log-loss is convex |
| Log-loss (per example) | $L = -y\log(f) - (1-y)\log(1-f)$ |
| Cost function | $J = \frac{1}{m}\sum_{i=1}^{m} L(f(x^{(i)}), y^{(i)})$ |
| Gradient (logistic) | Same form as linear: $\frac{\partial J}{\partial w_j} = \frac{1}{m}\sum(f(x^{(i)}) - y^{(i)})x_j^{(i)}$ |
| Regularized cost | Add $\frac{\lambda}{2m}\sum_{j=0}^{n-1} w_j^2$ to cost (L2/Ridge) |
| Feature mapping | Create polynomial features $x_1, x_2, x_1^2, x_1 x_2, \ldots$ to fit non-linear boundaries |

---

## Overfitting & Regularization

| Term | Definition |
|---|---|
| Underfitting (high bias) | Model too simple; fails to capture training pattern; $J_{train}$ high |
| Overfitting (high variance) | Model memorizes training data; $J_{train}$ low, $J_{cv}$ high |
| L2 regularization (Ridge) | Adds $\frac{\lambda}{2m}\sum w_j^2$ to cost; shrinks all weights toward zero |
| L1 regularization (Lasso) | Adds $\frac{\lambda}{2m}\sum |w_j|$ to cost; drives some weights to exactly zero (sparse) |
| $\lambda$ | Regularization parameter; $\lambda=0$ → no regularization, $\lambda \to \infty$ → all weights near zero |
| Fix high bias | More features, higher-degree model, decrease $\lambda$ |
| Fix high variance | More training data, fewer features, increase $\lambda$ |

---

## Neural Networks

| Term | Definition |
|---|---|
| Neuron | Computational unit: takes inputs, computes weighted sum + bias, applies activation |
| Layer | Group of neurons processing the same level of representation |
| Hidden layer | Any layer between input and output |
| Activation | Output of a neuron after applying an activation function |
| Sigmoid | $g(z) = \frac{1}{1+e^{-z}}$; use for binary classification output |
| ReLU | $g(z) = \max(0, z)$; preferred for hidden layers; avoids vanishing gradient |
| Linear | $g(z) = z$; use for regression output layers |
| Softmax | $a_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$; use for multi-class classification output |
| Forward pass | $a^{[l]} = g(W^{[l]}a^{[l-1]} + b^{[l]})$ |
| Binary cross-entropy | $L = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$ |
| Adam optimizer | Adaptive learning rates per parameter; more robust than vanilla gradient descent |
| Epoch | One full pass through the training dataset |

---

## Decision Trees

| Term | Definition |
|---|---|
| Entropy | $H(S) = -\sum_k p_k \log_2(p_k)$; measure of impurity (0 = pure node) |
| Information gain | $IG = H(\text{parent}) - [w_L H(L) + w_R H(R)]$; pick split with highest IG |
| Stopping criteria | Pure node, max depth reached, IG below threshold, node too small |
| One-hot encoding | Convert categorical feature with $k$ values into $k$ binary columns |
| Random Forest | Ensemble of trees trained on bootstrapped data + random feature subsets; reduces variance |
| XGBoost | Gradient-boosted trees; each tree corrects errors of the previous; strong baseline |

---

## Unsupervised Learning

| Term | Definition |
|---|---|
| Clustering | Group similar examples; no labels required |
| Anomaly detection | Flag examples with low probability under a learned distribution |
| Dimensionality reduction | Compress features while preserving structure (e.g. PCA) |
| K-means cost | $J = \frac{1}{m}\sum_{i=1}^{m}\|x^{(i)} - \mu_{c^{(i)}}\|^2$ (distortion) |
| Elbow method | Plot cost vs $K$; pick the "elbow" where adding more clusters yields diminishing returns |
| Gaussian density | $p(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ |
| Anomaly threshold | Flag $x$ as anomaly if $p(x) < \varepsilon$ |
| Collaborative filtering | Predict user–item ratings from patterns in the user–item matrix |
