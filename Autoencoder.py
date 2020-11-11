import random
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt

class Autoencoder(object):
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
    def training(self, training_data,lr, momentum):
        #Define beginning parameters
        batch_size = 35
        training_data = list(training_data)
        n = len(training_data)
        #Used for looking at the loss for each iteration
        loss = []
        js = []
        eLoss = 100
        j = 0
        avgLoss = []
        while(eLoss > .05):
            random.shuffle(training_data)
            mbs = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            for mb in mbs:
                avgLoss.append(self.updateParameters(mb, lr, momentum))
            eLoss = np.mean(avgLoss)
            # At every tenth epoch check loss
            if j % 10 == 0:
                loss.append(eLoss * 100)
                js.append(j)
            print("Epoch ", j)
            j += 1
            #Check for overfitting 
            if len(loss)  > 3:
                if (loss[-1] - eLoss * 100) < -0.5 and (loss[-2] - eLoss * 100) < -0.5:
                    print("Overfitting took place!")
                    return
        plt.plot(js, loss, '-r')
        plt.xlabel('Iterations') 
        plt.ylabel('Error Percentage (%)') 
        plt.title('Error for every 10 Iterations') 
        plt.show()
    
    #Updates the weights and biases based on backpropagation
    def updateParameters(self, mb, lr, momentum):
        biasGrad = [np.zeros(b.shape) for b in self.biases]
        weightGrad = [np.zeros(w.shape) for w in self.weights]
        losses = []
        for x, y in mb:
            #Feedforward
            activations = self.forward(x)
            losses.append(j2Loss(activations[-1], x))
            #Get the changes for each weight and bias
            deltaB, deltaW = self.backwardProp(activations, x)
            biasGrad = [nb+dnb for nb, dnb in zip(biasGrad, deltaB)]
            weightGrad = [nw+dnw for nw, dnw in zip(weightGrad, deltaW)]
        
        #Update weights and biases using backprop information along with momentum
        self.weights = [w-(lr/len(mb))*(momentum*nw) for w, nw in zip(self.weights, weightGrad)]
        self.biases = [b-(lr/len(mb))*(momentum*nb) for b, nb in zip(self.biases, biasGrad)]
        avgLoss = np.mean(losses)
        return avgLoss 
    
    #Backpropagation of the the activations and the actual (y)
    def backwardProp(self, activations, x):
        deltaB = [np.zeros(b.shape) for b in self.biases]
        deltaW = [np.zeros(w.shape) for w in self.weights]
        lastActivation = activations[-1]
        xValue = x
        cost = activations[-1] - x
        delta = cost * sigmoid_prime(activations[-1])
        deltaB[-1] = delta
        deltaW[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            sp = sigmoid_prime(activations[-l])
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            deltaB[-l] = delta
            deltaW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (deltaB, deltaW)
    
            
         
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

#J2 Loss Function
def j2Loss(output, x):
    loss = np.square(np.subtract(output,x)).mean() 
    return loss

#Grayscale Image Plotter to compare the input and the output
def TestgrayScaleImage(inputT, output):
    figure,axis = plt.subplots(nrows = 2, ncols = 8, figsize=(12,7))
    width,height = 28,28
    iterations = 0
    for val in axis.flat:
        xCounter = 0
        grid = [[0 for x in range(width)] for y in range(height)]
        for i in range(28):
            for j in range(28):
                if iterations < 8:
                    if inputT[iterations][xCounter][0] == 0.:
                        grid[j][i] = 0
                    else: 
                        grid[j][i] = inputT[iterations][xCounter][0]
                    xCounter += 1
                else:
                    if output[iterations-8][xCounter][0] == 0.:
                        grid[j][i] = 0
                    else: 
                        grid[j][i] = output[iterations-8][xCounter][0]
                    xCounter += 1
        iterations += 1
        color = plt.cm.get_cmap("Greys")
        newColor = color.reversed()
        val.imshow(grid,cmap = newColor)

    plt.tight_layout()
    plt.show()

#Create the bar graph plot for the error for each number
def plotBarGraphEachNum(testError, trainingError):
    labels = ['0','1','2','3','4','5','6','7','8','9']
    indices = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots()
    ax.bar(indices - width/2, testError[0], width, label = 'Test Set')
    ax.bar(indices+width/2, trainingError[0], width, label='Training Set')
    ax.set_ylabel('J2 Error (MSE) Percentage (%)')
    ax.set_title('J2 Error for Training and Test For Each Output')
    ax.set_xticks(indices)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.show()

#Create the bar graph plot for the overall error
def plotBarGraphOverall(testError, trainingError):
    indices = np.arange(1)
    width = .25
    fig, ax = plt.subplots()
    ax.bar(indices-width/2, testError, width, label='Test Set')
    ax.bar(indices+width/2, trainingError, width, label = 'Training Set')
    ax.set_ylabel('J2 Error (MSE) Percentage (%)')
    ax.set_title('J2 Error For Test and Training Set Overall')
    ax.get_xaxis().set_visible(False)
    ax.legend(loc='best')
    fig.tight_layout()
    plt.show()


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
#Append output size which is an image
size.append(28*28)

#Parameters
momentum = .9
learning_rate = .3

# #Define the network
NN = Autoencoder(size)
NN.training(trainingSet,learning_rate, momentum)

#Get training set error 
trainingSetError = np.zeros((10,1))
for x,y in trainingSet:
    activations = NN.forward(x)
    trainingSetError[getY(y)] += j2Loss(activations[-1], x)
for i in range(len(trainingSetError)):
    trainingSetError[i] = (trainingSetError[i] / 400)*100

#Get test set error
testSetError = np.zeros((10,1))
for x,y in testSet:
    activations = NN.forward(x)
    testSetError[getY(y)] +=  j2Loss(activations[-1], x)
for i in range(len(testSetError)):
    testSetError[i] = (testSetError[i] / 100)*100

plotBarGraphEachNum(testSetError, trainingSetError)
testSE = (np.sum(testSetError[0]))
trainSE = (np.sum(trainingSetError[0]))
plotBarGraphOverall(testSE, trainSE)

#Compare the input output for 8 random points in the test set
inputT = []
output = []
for i in range(8):
    numb = random.randint(0, 999)
    curX = testSet[numb][0]
    inputT.append(curX)
    activations = NN.forward(curX)
    output.append(activations[-1])
    
TestgrayScaleImage(inputT,output)
    
