# Neural Networks

## Overview

Neural networks are algorithms originally inspired by the brain. They learn complex, non-linear mappings by composing simple operations across many layers, making them powerful for tasks like image recognition, speech processing, and language modeling.

---

## Intuition

- **Neuron**: Takes 1…n inputs, computes a weighted sum plus bias, applies an activation function, and outputs 1…m values called **activations**.
- **Layer**: Consists of one or more neurons. The final layer is the **output layer**.
- **Neural network**: Multiple layers stacked together. All layers between input and output are called **hidden layers**.

**Example:** An input layer of 4 numbers feeds into a hidden layer of 3 neurons → 3 activation values → output layer emits the final prediction (also an activation).

---

## Forward Pass

For layer $l$:
$$\mathbf{a}^{[l]} = g\!\left(W^{[l]}\mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}\right)$$

where $g$ is the activation function, $W^{[l]}$ is the weight matrix, and $\mathbf{a}^{[0]} = \mathbf{x}$ (the input).

---

## TensorFlow / Keras Implementation

```python
layer_1 = Dense(units=3, activation="sigmoid")
layer_2 = Dense(units=1, activation="sigmoid")
model = Sequential([layer_1, layer_2])
```

---

## Training a Neural Network

### Step 1 — Define the model
Specify how to compute output given input $\mathbf{x}$ and parameters $W, b$:
```python
# Linear: f(x) = np.dot(w, x) + b
# Logistic: f_x = 1 / (1 + np.exp(-z))
```

### Step 2 — Specify loss and cost
```python
model.compile(optimizer='adam', loss=BinaryCrossentropy())
```

Binary cross-entropy loss:
$$L = -y\log(\hat{y}) - (1-y)\log(1-\hat{y})$$

### Step 3 — Train (gradient descent)
```python
model.fit(X, y, epochs=100)  # 100 iterations of backprop + parameter update
```

---

## Activation Functions

| Function | Formula | When to Use |
|---|---|---|
| **Sigmoid** | $g(z) = \frac{1}{1+e^{-z}}$ | Binary classification output layer |
| **ReLU** | $g(z) = \max(0, z)$ | Hidden layers (default choice); avoids vanishing gradient |
| **Linear** | $g(z) = z$ | Regression output layer (no activation) |
| **Softmax** | $a_j = \frac{e^{z_j}}{\sum_k e^{z_k}}$ | Multi-class classification output layer |

### How to Choose Activation Functions

- **Output layer — binary classification**: Sigmoid
- **Output layer — regression (any sign)**: Linear
- **Output layer — regression (non-negative, e.g. house price)**: ReLU
- **Hidden layers**: ReLU is the default; faster to train than sigmoid

---

## Multi-Class Classification

For $N$ possible output classes, the output layer uses **softmax regression**. Softmax outputs a probability distribution over all $N$ classes; the predicted class is $\arg\max_j a_j$.

---

## Adam Optimizer

Adam (Adaptive Moment Estimation) is better and faster than vanilla gradient descent:
- Adapts a separate learning rate for each parameter
- Combines momentum (exponentially weighted average of gradients) with RMSProp (squared gradient scaling)
- Default choice for training neural networks in practice

```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss=...)
```

---

## Debugging a Learning Algorithm

| Symptom | Likely Cause | Fix |
|---|---|---|
| $J_{train}$ high | High bias | More features, add polynomial features, decrease $\lambda$ |
| $J_{cv} \gg J_{train}$ | High variance | More training examples, fewer features, increase $\lambda$ |

**Specific strategies:**

- **Get more training examples**: Helps high variance; has no effect on high bias.
- **Try smaller set of features**: Simplifies model, reduces high variance.
- **Try getting additional features**: Adds expressiveness, reduces high bias.
- **Add polynomial features**: Fixes high bias.
- **Decrease $\lambda$**: Fixes high bias.
- **Increase $\lambda$**: Fixes high variance.

---

## Practical Notes

- Always scale/normalize inputs before training (zero mean, unit variance).
- Use ReLU in hidden layers as the default; sigmoid is mainly for output layers.
- Monitor the loss curve over epochs — it should decrease smoothly. Spikes suggest too-high learning rate.
- Prefer Adam over plain gradient descent for most practical work.

---

## Interview Quick-Fire

**Q: What is the vanishing gradient problem?**
During backpropagation through many layers, sigmoid/tanh activations squash gradients toward zero, making early layers learn very slowly. ReLU mitigates this because its gradient is 1 for positive inputs.

**Q: Why use ReLU instead of sigmoid in hidden layers?**
ReLU is computationally cheap, does not saturate for large positive values (no vanishing gradient), and trains faster. Sigmoid saturates at both ends, causing vanishing gradients.

**Q: What does softmax do?**
Converts raw logits into a probability distribution over $N$ classes: $a_j = e^{z_j} / \sum_k e^{z_k}$. All outputs sum to 1.

**Q: What is Adam and why is it preferred?**
Adam adapts the learning rate per parameter using estimates of first and second moments of gradients. It converges faster and is less sensitive to learning rate choice than SGD.

**Q: How do you diagnose high bias vs high variance?**
Compare $J_{train}$ and $J_{cv}$. Both high → high bias. $J_{train}$ low but $J_{cv}$ much higher → high variance.
