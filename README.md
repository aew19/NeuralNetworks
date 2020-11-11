# NeuralNetworks

### Programs Included:
- NeuralNetwork.py
- Autoencoder.py 

#### Neural Network:

This file is a feedforward neural network which is used to learn the MNIST dataset and then be able to classify the dataset.

A few key features:
- The user can input the number of hidden layers they want and the number of neurons the want at each hidden layer
- Stochastic Gradient Decent for the backpropagation. Where the training set is split into mini batches and sent to the backpropagation method to learn. 
- Every 10 epochs the error is found and saved for a graph. The error is 1 - the balanced accuracy of the training set
- After training the test set and the training set are sent in and analyzed using a confusion matrix 

Output:

1. An error graph of the training set:
2. A confusion matrix for the test set:
3. A confusion matrix for the training set:

#### Autoencoder:

The file is a feedforward neural network that takes the input datapoint and tries to reproduce that input in dataset. So basically it encodes the data and then decodes it.

A few key features:
- The user can input the number of hidden layers they want and the number of neurons the want at each hidden layer
- Stochastic Gradient Decent for the backpropagation. Where the training set is split into mini batches and sent to the backpropagation method to learn. 
- Every 10 epochs the error is found and saved for a graph. This uses the J2 loss function for the error which is the Mean Squared Error. 
- After training both the training and test set are sent in and the Mean Squared Error is looked at overall and then a breakdown for each number is looked at as well.
- The input vs the output are also compared for the training set using a visual

Output:

1. An error graph comparing the training and test set error after training
2. An error graph comparing the training and test set error for each number after training 
3. During training error 
4. A comparision of the input and output visual 

#### How to run:

1. pip install -r requirements.txt
2. python NeuralNetwork.py
3. python Autoencoder.py

** I used python version 3
