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
![image](https://user-images.githubusercontent.com/35077827/98849967-0cf80180-2422-11eb-8dbb-2447a368bffb.png)
2. A confusion matrix for the test set:
![image](https://user-images.githubusercontent.com/35077827/98850108-416bbd80-2422-11eb-8d2f-44a5da943959.png)
3. A confusion matrix for the training set:
![image](https://user-images.githubusercontent.com/35077827/98850172-58121480-2422-11eb-9432-85c96f4e0793.png)

#### Autoencoder:

The file is a feedforward neural network that takes the input datapoint and tries to reproduce that input in dataset. So basically it encodes the data and then decodes it.

A few key features:
- The user can input the number of hidden layers they want and the number of neurons the want at each hidden layer
- Stochastic Gradient Decent for the backpropagation. Where the training set is split into mini batches and sent to the backpropagation method to learn. 
- Every 10 epochs the error is found and saved for a graph. This uses the J2 loss function for the error which is the Mean Squared Error. 
- After training both the training and test set are sent in and the Mean Squared Error is looked at overall and then a breakdown for each number is looked at as well.
- The input vs the output are also compared for the training set using a visual

Output:

1. An error graph comparing the training and test set error after training:
![image](https://user-images.githubusercontent.com/35077827/98850233-70822f00-2422-11eb-998c-4dd58b5b64b5.png)
2. An error graph comparing the training and test set error for each number after training:
![image](https://user-images.githubusercontent.com/35077827/98850265-8132a500-2422-11eb-9b73-aafb9f873ccf.png)
3. During training error using Mean Squred Loss:
![image](https://user-images.githubusercontent.com/35077827/98850312-94de0b80-2422-11eb-9c70-3f416fb517ce.png)
4. A comparision of the input and output visual:
![image](https://user-images.githubusercontent.com/35077827/98850453-d1116c00-2422-11eb-8001-3609893e3205.png)

#### How to run:

1. pip install -r requirements.txt
2. python NeuralNetwork.py
3. python Autoencoder.py

** I used python version 3
