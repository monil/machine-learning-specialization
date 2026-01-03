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

