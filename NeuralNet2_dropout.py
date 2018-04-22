#NeuralNet2_dropout - why the '2' at the end.  The original implimentation
#was a nice OO based class with Node classes, layer classes, NeuralNetwork Classes
#and the math was a mess to perform (object indexing hell)
#The neuron-by-neuron view is messy to think about

#For making the math easier let us adopt a matrix based approach


#Commment 
#Idea was to make a simple neural network that worked first, then
#add features as time permits.
#Hence, quadratic loss, Sigmoid activation output layer
#Stick closely to MNIST example to start with, then
#add features
#
# Stochastic gradient descent
# Why? as before simplicity
# easiest to implement.  For every training example update the model
# but more computatinally expensive, noisy, harder to settle on a min
#
# With more time would  have probably moved to mini-batch
# more efficient, but less vulnerable to finding local min as batch decent
#
# Quadratic cost
# simplicity
# issues - bug in calculateding the cost when doing the test.  Not summing
# Test over entire test set, but doing SGD one at a time.
#
# Should move to using cross-entropy loss rather than quadradic loss
# non-negatice check
# neuron is close to desired value give value cclose to zero check
# rate of learing is controlled by difference between output and target, error of the output
#
# Softmax - Probably a better loss function, might solve the issue of the increasing cost
# And increase in one output value depresses the other, where it seems using the quadratic cost
# only minimises over all outputs
# push values down for other outputs
#
# Why is the cost increasing on the test set?
# looking at running over test data, the correct value is getting larger, but so do the others
# not getting supressed to smaller values
# hence the larger loss
# probably overfitting

#Dropout - ant small chnages in the imput not to make large chnages in the output, slow gradual function of input
# Network with large weights might chnage the output a large amount, large weights that carry a lot of info about the noise
# Want to constrain the weight, keep them from growing quickly
# Basically we train the network on many different networks rather than 1 network
# to differnt networks will overfit, but the results has to try and give the right value
# Neurons cannot co-adapt with each other, because at different times they are on and off and cannot reply on each other.
#
# inverse dropout
# Leaves the test phase neurons untouched, no scaling at test time.


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#Comment
#two functions - original plan was to add the ability to use a variety of
#output activation functions
#Start with Sigmoid such as the MNSIT example given by the
#neural network website
#But for classification, softmax + cross entropy loss would be the more normal choice

def Sigmoid(x,derivative=False):
    if(derivative):
        #return x*(1.0-x)
        return (1.0-Sigmoid(x))*Sigmoid(x)
    else:
        return  1.0/(1.0 + np.exp(-x))

def Softmax(x):
    z = np.sum(np.exp(x), axis=1)
    z = z.reshape(z.shape[0], 1)
    return np.exp(x) / z
            
#Comment
#Seperate class for layers in order to keep things a bit more clear and clean
#Nodes in a layer are just vectors in this setup

class layer:
    def __init__(self,size,activation=Sigmoid,inputLayer=False, outputLayer=False):
        self.activation = activation
        self.inputLayer = inputLayer
        self.outputLayer = outputLayer
        self.hiddenLayer = False
        if(not outputLayer and not inputLayer):
            self.hiddenLayer = True
            
        #define output and input vector
        self.Z = np.zeros((size[0],1))
        self.A = np.zeros((size[0],1))
        self.I = np.zeros((size[1],1))
        self.DO = np.zeros((size[0],1)) # for the dropout mask
        
        # matrix for the weights
        self.W = None
        #Matrix to store the update to the W matrix
        self.Wup = None
        # Save the derivatives of the activation function
        self.Fprime = None
        # Save the deltas for this layer
        self.D = None
        #Bias vector
        self.B = None
            
        self.W = np.random.randn(size[0],size[1])
        self.B = np.random.randn(size[0],1)
            
        self.Fprime = np.zeros((size[0],1))
        
    def feedForward(self):
        #Feed the inputs through the layer
        #do product between the inputs and the weight matrix for
        #this layer, then apply activation function

        if(self.inputLayer):
            #We don't feed the input layer though
            #so this should not be called
            return self.A
            
        #else get the activations from the previous layer
        #and assign to the node outputs
        self.Z = np.dot(self.W,self.I)+self.B
        self.A = self.activation(self.Z)
        self.Fprime = self.activation(self.Z,derivative=True)
        return self.A


#Comment, a simple Neural net.
#Started to add features to allow multiple activation functions
#Added dropout to give it a try

#Dropout -  
            
class neuralNet2:
    def __init__(self,layers,dropout = False):
        #set the random seed
        np.random.seed(42)
        self.dropout = dropout
        #layers is an array of the length equal to the number of
        #layers (including the input and output layers)
        #Each entry is the number of nodes
        self.layerSize = layers
        self.nLayers = len(self.layerSize)
        self.layers = []
    
        #define the layers for input, output and hidden
        for idx in range(self.nLayers):
            if(idx == 0):
                #input layer
                 self.layers.append(
                    layer(size=[self.layerSize[idx],self.layerSize[idx]],
                          inputLayer=True))
            elif(idx == self.nLayers - 1):
                #output layer 
                self.layers.append(
                    layer(size=[self.layerSize[idx],self.layerSize[idx-1]],
                          outputLayer=True,
                          activation=Sigmoid))
            else:
                #hidden layer
                 self.layers.append(
                    layer(size=[self.layerSize[idx],self.layerSize[idx-1]],
                          activation=Sigmoid))
            

    def feedForward(self,data):
        #We need to set the activations of the input layer
        #to pass into the 2nd layer (hidden or output layer)
        data = data.reshape((data.shape[0],1))

        #Setup the input node to have the correct output values
        self.layers[0].I = data
        self.layers[0].Z = data
        self.layers[0].A = data
        
        #loop over layers to feed these values forward through the output layer
        for idx in range(1,self.nLayers):
            #set the output of the previous layer to the input of the next layer
            #Feed forward will use the input of the previous layer and
            #calculate the outputs
            if(self.dropout and self.layers[idx-1].hiddenLayer is True):
                self.layers[idx].I = self.layers[idx].I * self.layers[idx].DO
            self.layers[idx].I = self.layers[idx - 1].feedForward()            
            #if dropout, apply dropout mask to the input vector
            #this should only happen during training

                
        #return the output of the output layer
        return self.layers[-1].feedForward()
    
    def backPropagate(self, target, eta):
        target = target.reshape((target.shape[0],1))
        #We will use the quadratic cost
        #start by calculating the difference for the output layer 
        self.layers[-1].D = (self.layers[-1].A - target)*self.layers[-1].Fprime
        self.layers[-1].Wup = np.dot(self.layers[-1].D, self.layers[-2].A.transpose())
        #loop over the hidden layers
        for i in range(2, self.nLayers):
            self.layers[-i].D = np.dot(self.layers[-i+1].W.transpose(), self.layers[-i+1].D)*self.layers[-i].Fprime
            if(self.dropout and self.layers[-i].hiddenLayer is True):
                self.layers[-i].D = self.layers[-i].D*self.layers[-i+1].DO
            self.layers[-i].Wup = np.dot(self.layers[-i].D, self.layers[-i-1].A.transpose())

        #update weights and biases
        for i in range(1, self.nLayers):
            self.layers[i].W = self.layers[i].W - eta*self.layers[i].Wup
            self.layers[i].B = self.layers[i].B - eta*self.layers[i].D
            
        return (self.layers[-1].A - target)

    def SetDropout(self,prob):
        #Set the dropout masks for each of the hidden layers
        for i in range(0,self.nLayers):
            if(self.layers[i-1].hiddenLayer is True):
                #divide by dropout probability as we are doing
                #inverse dropout so that testing is unchanged
                #size of this layers drop out mask (to mask out the nodes inputing into
                #this layer) should be the size of the output vector of the previous
                #layer
                self.layers[i].DO = np.random.binomial(1,1.0 - prob,size=self.layers[i-1].A.shape) / (1.0 - prob)
                
    def Train(self, data, target, test=None, testTarget=None, epochs=1000,
              eta=0.05,acc=0.95,dropout=0.5):
        #loop over the numbr of epochs
        numEpochs = 0
        accuracy = 0
        accuracyEpoch = []
        lossEpoch = []
        #loop until the number of epochs or desired test accuracy is reached
        while(numEpochs < epochs and accuracy < acc):
            #implement dropout
            #each epoch will get a new dropout array for each hidden layer
            if(self.dropout):
                self.SetDropout(dropout)
                
            self.costEpoch = 0
            for j,dataInput in enumerate(data):
                output = self.feedForward(dataInput)
                diff = self.backPropagate(target[j],eta)

            #After the epoch is finshed we should test using the test
            #data set.  Get the loss and the accuracy
            if(test.any()):
                #if dropout is one turn off for testing, and then
                #re-enable
                restoreDropout = False
                if(self.dropout):
                    self.dropout = False
                    restoreDropout = True

                accuracy,loss = self.Test(test,testTarget)
                accuracyEpoch.append(accuracy)
                lossEpoch.append(loss)
                print "Epoch ",numEpochs," accuracy ",accuracy*100," loss ",loss
                #restore dropout if needed
                if(restoreDropout):
                    self.dropout = True
                    restoreDropout = False
            numEpochs = numEpochs + 1    
        return accuracyEpoch,lossEpoch
    
    def Test(self,test,target):
        #feed our data forward
        #Get the output
        #compare to the target
        #return the accuracy as a percent
        count = 0;
        loss = 0
        for j,testData in enumerate(test):
            
            output = self.feedForward(testData)
            #check if the output is correct by finding the output node with
            #the largest value
            if(np.argmax(self.layers[-1].A) == np.argmax(target[j])):
                count += 1
                #        loss = 0.5*np.linalg.norm(output)*np.linalg.norm(output)
            loss += 0.5*np.linalg.norm(output)**2
            print self.layers[-1].A, target[j], np.linalg.norm(output)
        loss = loss/test.shape[0]

        
        return 1.0*count/test.shape[0],loss

    def Validate(self,validate,target):
        #Validation
        return self.Test(validate,target)

    def Predict(self,data):
        output = None
        for i,data in enumerate(data):
            self.feedForward(data)
            if(i == 0):
                output = self.layers[-1].A.transpose()
            else:
                output = np.hstack([output,self.layers[-1].A.transpose()])

        output = output.reshape(-1,self.layerSize[-1])
        return output
    
if __name__ == "__main__":
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_norm = preprocessing.normalize(X)
    #We want to split this data 3 ways
    #60% for training
    #20% for testing
    #20% for validation
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
    X_test2, X_val,y_test2,y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=5)
    #Now we massage the data into the format required ny the Neural Netork code 
    #iris.data is in a format that we need, but the output data needs to be in the
    #same layout
    Setosa = np.array([1,0,0])
    Versicolor = np.array([0,1,0])
    Virginica = np.array([0,0,1])

    #beware, some terrible data munging occurs below
    y_train2 = None
    y_test3 = None
    y_val2 = None
    for i,line in enumerate(X_train):
        #choose the right target
        target = None
        if(y_train[i] == 0):
            target = Setosa
        elif(y_train[i] == 1):
            target = Versicolor
        else:
            target = Virginica
        if(i == 0):
            y_train2 = target
        else:
            y_train2 = np.hstack([y_train2,target])
    y_train2 = y_train2.reshape(-1,3)
    
    for i,line in enumerate(X_test2):
        #choose the right target
        target = None
        if(y_test2[i] == 0):
            target = Setosa
        elif(y_test2[i] == 1):
            target = Versicolor
        else:
            target = Virginica
        if(i == 0):
            y_test3 = target
        else:
            y_test3 = np.hstack([y_test3,target])
    y_test3 = y_test3.reshape(-1,3)

    for i,line in enumerate(X_val):
        #choose the right target
        target = None
        if(y_val[i] == 0):
            target = Setosa
        elif(y_val[i] == 1):
            target = Versicolor
        else:
            target = Virginica
        if(i == 0):
            y_val2 = target
        else:
            y_val2 = np.hstack([y_val2,target])
    y_val2 = y_val2.reshape(-1,3)


    #At this point the training data in the correct format is:
    #X_train,y_train2
    #testing:
    #X_test2,y_test3
    #validation:
    #X_val,y_val2

    #an example neural net with 4 inputs (ie. iris dataset inputs), 3 outputs
    # for each of the flower types and 2 hidden layers of
    #5 and 3 nodes

    #50% dropout
    NN_drop = neuralNet2([4,5,5,3],dropout=True)
    #Train the network, check loss and accuracy after each epoch
    accuracy_drop,loss_drop = NN_drop.Train(X_train,y_train2,X_test2,y_test3,epochs=10000,eta=0.001,acc=0.99,dropout=0.5)
    #run over the validation dataset
    print "Running over the validation set:\nValidation Accuracy %f loss %f\n"%NN_drop.Validate(X_val,y_val2)
    
    #No dropout
    NN = neuralNet2([4,5,3],dropout=False)
    #Train the network, check loss and accuracy after each epoch
    accuracy,loss = NN.Train(X_train,y_train2,X_test2,y_test3,epochs=1000,eta=0.02,acc=0.99)
    #run over the validation dataset
    print "Running over the validation set:\nValidation Accuracy %f loss %f\n"%NN.Validate(X_val,y_val2)

    #plot the accuracy and loss
    p1 = plt.subplot(2,1,1)
    plt.plot(accuracy_drop)
    plt.title("Accuracy vs. epochs (50% dropout)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    p2 = plt.subplot(2,1,2)
    plt.plot(loss_drop)
    plt.title("Mean Squared Loss vs. epochs (50% dropout)")
    plt.ylabel("Mean Squared Loss")
    plt.xlabel("Epochs")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig("acc_loss_with_dropout.pdf")
    plt.clf()
    
    p3 = plt.subplot(2,1,1)
    plt.plot(accuracy)
    plt.title("Accuracy vs. epochs")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    p3 = plt.subplot(2,1,2)
    plt.plot(loss)
    plt.title("Mean Squared Loss vs. epochs")
    plt.ylabel("Mean Squared Loss")
    plt.xlabel("Epochs")

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig("acc_loss.pdf")
    
