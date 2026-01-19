Neural networks Intuition 

- Algorithms that started to mimic the brain 

- Neuron - Takes 1..n inputs applies the model and then outputs 1..m results called **activations**

- Layer consists of one or more neurons as above. Final layer is the output layer. 

- Neural networks is formed by multiple layer as mentioned above

Eg: Input layer consisting for 4 numbers goes to first layer consisting of 3 neurons which outputs 3 numbers called activation values of layer 1 and is fed to output layer which emits final output also called as activation of output layer.


- All layer between input and output layer are called hidden layers

=================================================================

Use TensorFlow for Inference in code with Neural Networks. Tensor flow and numpy has their way of representing a matrix.

Eg: to create a neural network in Tensorflow

    layer_1 = Dense(units=3, activation="sigmoid")
    layer_2 = Dense(units=1, activation="sigmoid")
    model = sequential([layer_1, layer_2])

Check Sigmoid curve

==================================================================

Training a neural network steps

1. Specify how to compute o/p given input x and parameters w,b (Define a model)

eg: fw_b(x) = w*x + b 
            = np.dot(w,x) + b

    f_x = 1 / (1 + np.exp(-z))

2. Specify loss and cost 

    model.compile(X, y, loss=BinaryCrossEntropy())

3. Train on data to minimize Jw_b ( Gradient Descent)

    model.fit(X,y, epochs=100) - 100 iterations of loss function 

====================================================================

Diffferent Types of activation functions for neurons, used largely in neural networks

1. Sigmoid : Emits o/p either 0 or 1
2. ReLU (Rectified Linear Unit) : g(z) = max (0, z)
3. Linear activation unit : g(z) = z , Also called as no activation function


How to choose activation function 

Binary Classification : Use sigmoid activation function

Regression (eg: predicted stock price) : Linear activation function

Regression (ReLU) only positive (eg: Predicting house prices) : 


====================================================================

Multiclass classification (N possible outputs)

Output layer uses softmax regression 


=====================================================================

Adam optimization algorithm - Better than gradient descent and faster - used as a default for neural networks. 

===================================================================

Debugging learning algorithm 

- Get More training examples
  If high bias increasing training data has no effect. If high variance increasing training data does help 

- Try smaller set of features 
  Makes model simpler and reduces high variance problem 

- Try getting additional features 
  Makes model flexible reduces high bias problems

- Try adding polynomial features 
  Fixes high bias 

- Try decrasing lamda
  Fixes high bias 

- Try increasing lambda
  Fixes high variance

High Bias - Unable to fit in more training set 

High Variance - Unable to fit in more test set