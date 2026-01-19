# Machine Learning — Key Concepts (Cheat Sheet)

**Purpose**: Quick textual definitions of major concepts for rapid revision.

**Supervised Learning**: A learning paradigm where a model is trained on labeled examples (input paired with correct output). Use it when you have historical input–output pairs and want the model to predict outputs for new inputs.

**Unsupervised Learning**: A learning paradigm that finds structure in unlabeled data (clustering, dimensionality reduction, density estimation). Use it when you have data without labels and want to discover groups, features, or latent structure.

**Linear Regression**: A parametric model that predicts a continuous target as a weighted sum of input features (plus bias). Use it for regression tasks with approximately linear relationships and when interpretability and simplicity are priorities.

**Logistic Regression**: A linear model passed through a sigmoid to predict class probabilities; yields binary (or multi-class) classification. Use it for classification problems where the decision boundary is roughly linear and probability estimates are useful.

**Cost / Loss Function**: A scalar function that measures how well the model's predictions match targets (e.g., mean squared error for regression, logistic loss for classification). Learning minimizes this function to find model parameters.

**Gradient Descent**: An iterative optimization method that updates model parameters in the direction opposite the gradient of the cost function, controlled by a learning rate. Use it to minimize differentiable loss functions when closed-form solutions are impractical.

**Decision Boundary**: The surface (line, plane, or hypersurface) in feature space that separates predicted classes. It is defined by the set of points where the model’s predicted class probability equals a threshold (often 0.5).

**Overfitting**: When a model fits training data too closely (including noise) and performs poorly on unseen data; typically caused by excessive model complexity or insufficient data.

**Regularization**: Techniques that penalize model complexity (e.g., L1 or L2 penalties) or constrain training (dropout, early stopping) to reduce overfitting and improve generalization.

**Bias**: High Bias (underfit). Jtrain is high and Jcv is high. 

**Variance**: overfit. Jtrain is low & Jcv is very high. 

=============================================================


  
  