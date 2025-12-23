Eg:

Spam Filtering
Speech Recognition
machine transltion
online advertising
self-driving car
visual inspection

Works on input X and produce output Y

1. Regression 

==> x = y + mc

Eg: Housing price prediction by plotting on x and y axis

- predict a number 

2. Classification 

Eg: Breast Cancer Detection 

- Only 2 possible outputs ( benign and malignant )
- Can Have more than multiple classes
- small number of possible outputs as contrast to Regression
- 2 or more inputs possible 


Regression Algorithms

a. Linear Regression

    x = "input" variable
        feature
    y = "output" variable
        "target" variable

    Training Set = (x(i), y(i))

    f(x) = wx + b -- also called as model

    Univariate Linear Regression - one variable x 
    
=======================================================================================

Cost function : Funtion to compute values of parameters (w,b) in linear regresssion. 

Goal is to minimize cost function for high model accuracy 

MSE= 1/m
  
            i=1
            ∑
            m
​
# Supervised Learning — Notes

Overview
-------

Supervised learning maps inputs X to outputs Y using labeled training examples. Common applications:

- Spam filtering
- Speech recognition
- Machine translation
- Online advertising (click prediction)
- Self-driving cars
- Visual inspection / quality control

Problem types
-------------

1. Regression

- Predicts a continuous value (e.g., house price).
- Example: housing price prediction — plot features on the x axis and the target price on the y axis.

2. Classification

- Predicts a discrete class label (e.g., benign vs malignant).
- Can be binary (2 classes) or multiclass (more than 2 classes).

Regression algorithms
---------------------

### Linear Regression

- Input (feature): $x$ (or $x^{(i)}$ for example i)
- Output (target): $y$ (or $y^{(i)}$)
- Training set: $(x^{(i)}, y^{(i)})$ for $i=1\\dots m$
- Model (univariate):
  $$f(x)=w x + b$$

Cost function
---------------

We measure model quality using a cost (loss) function and seek parameters that minimize it. A common choice for linear regression is the mean squared error (MSE):

$$
J(w,b)=\\frac{1}{2m}\\sum_{i=1}^{m}\\bigl(f(x^{(i)})-y^{(i)}\\bigr)^2
$$

The factor $\\tfrac{1}{2}$ is conventional because it simplifies gradients.

Gradient Descent
-----------------

To minimize $J(w,b)$ we use gradient descent: repeatedly update parameters in the direction of the negative gradient until convergence.

Update rules (batch gradient descent):

$$
w := w - \\alpha\\ \\frac{1}{m}\\sum_{i=1}^m\\bigl(f(x^{(i)})-y^{(i)}\\bigr)\\,x^{(i)}
$$

$$
b := b - \\alpha\\ \\frac{1}{m}\\sum_{i=1}^m\\bigl(f(x^{(i)})-y^{(i)}\\bigr)
$$

where $\\alpha$ is the learning rate.

Next steps / notes
------------------

- Extend to multivariate linear regression where $x$ is a vector and $w$ is a weight vector.
- Regularization (e.g., Ridge/L2, Lasso/L1) helps reduce overfitting.
- For classification tasks consider logistic regression, decision trees, SVMs, and neural networks.

---
