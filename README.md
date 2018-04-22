The program NeuralNet2 implements a simple Feedforward Neural Network with inverse dropout
(the masks which disables the hidden layer nodes are scaled by the dropout probability)
in order to eliminate the need for modifiying the running of the test data after each epoch.
Sigmoid activations are used for all layers (though the layer class can support multiple
activation functions) to simplify the implementation.  For classification input data
should be in the form [0,1,0,0,..0] corresponding to each output node.  The output node
the max value is the prediction.

The iris data set was used for training, testing and validation
(two pdf flies are generated for the accuracy and loss plots of the test data
after every epoch).  The data was split 60% training, 20% testing ater each epoch,
and 20% validation.  The data was randomised to ensure that the neural network is
on all features.

2 tests are done, one with no dropout and a second with 50% dropout.

==================================

To create a new object the usage is:
def neuralNet2(layers,dropout=False)

Initialises the neural network.  At this moment only Sigmoid activation is
available, though the layer class can support the use of multiple
activation functions

layers - a python list specifying the number of nodes in each layer
eg. [4,5,3] specifies a 4 node input layer, 5 node hidden layer and
a 3 node output layer

dropout - True or False, specifies whether dropout is to be used

=================================

Training the network:

def Train(self, data, target, test=None, testTarget=None, epochs=1000,
              eta=0.05,acc=0.95,dropout=0.5)

returns a list for accuracy and loss calculated from the test dataset for each epoch


data - should be a numpy array with shape (-1,M) where M is the number of input nodes
[[A1,A2,A3...,AN],
.
.
.
[Z1,Z2,Z3...,ZN]]

target - see above for format, should be same length as the input data

test - test data to be run after every epoch.  Same format as the training dataset

testTarget - test data's target.  Same length as the test data

epochs - max number of epochs

eta - learning rate

acc - accuracy that if reached will stop training, even if the max number of epochs
has not been reached

dropout - percentage og hidden layers to drop for each epoch



====================================

def Validate(self,validate,target)

returns the accuracy and loss of the validation data 

validate - dataset for validation.  Same format as the training step

target -  target data for the vlidation data


=====================================


def Predict(self,data)

Given data returns the predicted output from the neural net.

if being used for catagorisation such as with the iris dataset the
max output node should be used to make the prediction