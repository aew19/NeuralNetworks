import random
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt

class FeedforwardNetwork(object):
    #Intalize our network
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    #Feedforward for the network, takes in point x and returns activation for all layers
    def forward(self, a):
        activation = a
        activations = [a] 
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], activation) + self.biases[i]
            activation = sigmoid(z)
            activations.append(activation)
        return activations
    
    #Main function for training takes in all parameters and data and then outputs an evaluation
    def training(self, training_data, epochs, lr, momentum):
        batch_size = 35
        training_data = list(training_data)
        n = len(training_data)
        errors = []
        js = []
        for j in range(epochs):
            if j % 10 == 0:
                numRight = self.compare(training_data)
                numWrong = n - numRight
                bA = balancedAccuracy(numRight, numWrong)
                errors.append((1 - bA) * 100)
                js.append(j)
                #Check for overfitting 
                if len(errors)  > 3:
                    if (errors[-1] - (1 - bA) * 100) < -0.5 and (errors[-2] - (1 - bA) * 100) < -0.5:
                        print("Overfitting took place!")
                        return
            random.shuffle(training_data)
            mbs = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for mb in mbs:
                self.updateParameters(mb, lr, momentum)
            print("Epoch ", j)

        plt.plot(js, errors, '-r')
        plt.xlabel('Epochs') 
        plt.ylabel('Error Percentage (%)') 
        plt.title('Error for every 10 epoch') 
        plt.show()
    
    #Updates the weights and biases based on backpropagation
    def updateParameters(self, mb, lr, momentum):
        biasGrad = [np.zeros(b.shape) for b in self.biases]
        weightGrad = [np.zeros(w.shape) for w in self.weights]

        for x, y in mb:
            #Feedforward
            activations = self.forward(x)
            #Get the changes for each weight and bias
            deltaB, deltaW = self.backwardProp(activations, y)
            biasGrad = [nb+dnb for nb, dnb in zip(biasGrad, deltaB)]
            weightGrad = [nw+dnw for nw, dnw in zip(weightGrad, deltaW)]
        
        #Update weights and biases using backprop information along with momentum
        self.weights = [w-(lr/len(mb))*(momentum*nw) for w, nw in zip(self.weights, weightGrad)]
        self.biases = [b-(lr/len(mb))*(momentum*nb) for b, nb in zip(self.biases, biasGrad)]
      
    
    #Backpropagation of the the activations and the actual (y)
    def backwardProp(self, activations, y):
        deltaB = [np.zeros(b.shape) for b in self.biases]
        deltaW = [np.zeros(w.shape) for w in self.weights]
        cost = activations[-1] - y
        delta = cost * sigmoid_prime(activations[-1])
        deltaB[-1] = delta
        deltaW[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            sp = sigmoid_prime(activations[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            deltaB[-l] = delta
            deltaW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (deltaB, deltaW)
    
    #Evaluates the results based on current weights
    def compare(self, data):
        correct = 0
        for x,y in data:
            activation = self.forward(x)[-1]
            results = getMaxPredValue(activation)
            if (results == getY(y)):
                correct += 1
        return correct
    
            
         
#Returns the sigmoid for the activation
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

#Gets derivative of sigmoid function 
def sigmoid_prime(activation):
    prime = activation
    for i in range(len(activation)):
        prime[i] = activation[i] * (1.0 - activation[i])
    return prime

#Get the actual Y value
def getY(y):
    for i in range(len(y)):
        if y[i] == 1.0:
            return i

#Parse data into an array
def parse():
    fileArray = []
    files = open("MNISTnumImages5000_balanced.txt", "r")
    for i in files:
        removeNL = i.rstrip('\n')
        j = [float(k) for k in removeNL.split('\t')]
        fileArray.append(j)
    return fileArray

#Parse labels into an array
def parseLabel():
    labelArray = []
    files = open("MNISTnumLabels5000_balanced.txt", "r")
    for i in files:
        labelArray.append(int(i.rstrip('\n')))
    return labelArray

#Randomly picks 4000 data points for training set and the rest for test set
#Stored in a tuple (input, expectedOutput)
def getData():
    lines = parse()
    labelLines = parseLabel()
    inSet = []
    notInSet = []
    trainx = []
    trainy = []
    testx = []
    testy = []

    trainingSet = []
    testSet = []
    
    for i in range(5000):
        notInSet.append(i)
    j = 400
    i = 0
    k = 499
    while len(trainx) < 4000:
        while len(trainx) < j:
            value = randint(i,k)
            if value not in inSet:
                inSet.append(value)
                notInSet.remove(value)
                trainx.append(lines[value])
                trainy.append(labelLines[value])   
        j = j + 400
        i = k + 1
        k = k + 500
    #Get the test set
    for i in notInSet:
        testx.append(lines[i])
        testy.append(labelLines[i])
    
    trainx = [np.reshape(x, (784, 1)) for x in trainx]
    trainy = [hotCodeY(y) for y in trainy]
    trainingSet = list(zip(trainx, trainy))
    testx = [np.reshape(x, (784, 1)) for x in testx]
    testy = [hotCodeY(y) for y in testy]
    testSet =list(zip(testx, testy))
    return (trainingSet, testSet)

#Hot codes so if 0 the value will be [1,0,0,0,0,0,0,0,0,0]
def hotCodeY(j):
    e = np.zeros((10,1), dtype=int)
    e[j] = 1
    return e

def getMaxPredValue(ys):
    number = 0
    maxVal = ys[0]
    for i in range(len(ys)):
        if ys[i] > maxVal:
            maxVal = ys[i]
            number = i
    return number

#Gets the balanced accuracy 
def balancedAccuracy(right, wrong):
    bA =  right / (right + wrong)
    return bA


numberHLtxt = input("Please Enter the Number of Hidden Layers: ")
numberHNtxt = input("Please Enter the Number of Hidden Neurons: ")
numberHL = int(numberHLtxt)
numberHN = int(numberHNtxt)

trainingSet = []
testSet = []
size = []

trainingSet, testSet = getData()

#Append input size
size.append(28*28)
for i in range(numberHL):
    #Append all of the hidden layers
    size.append(numberHN)
#Append output size
size.append(10)

#Parameters
epoch = 500
momentum = .9
learning_rate = .3

#Define the network
NN = FeedforwardNetwork(size)
NN.training(trainingSet,epoch,learning_rate, momentum)

prediction = []
actual = []

#Create the confusion matrix from the test set
print("Testing Confusion Matrix")
for x, y in testSet:
    prediction.append(getMaxPredValue(NN.forward(x)[-1]))
    actual.append(getY(y))
actualSeries = pd.Series(actual, name='Actual')
predictionSeries = pd.Series(prediction, name='Predicted')
testConfusion = pd.crosstab(actualSeries, predictionSeries, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(testConfusion)

#Create the confusion matrix from the training set
print("Training Confusion Matrix")
prediction = []
actual = []
for x, y in trainingSet:
    prediction.append(getMaxPredValue(NN.forward(x)[-1]))
    actual.append(getY(y))
actualSeries = pd.Series(actual, name='Actual')
predictionSeries = pd.Series(prediction, name='Predicted')
trainingConfusion = pd.crosstab(actualSeries, predictionSeries, rownames=['Actual'], colnames=['Predicted'], margins=True)
print(trainingConfusion)
    
