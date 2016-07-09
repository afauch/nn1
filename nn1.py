import csv
import numpy as np
from numpy import linalg
from scipy import optimize
from pylab import *

# Step 0: Bring in data from csv

# Step 1
# Create array of data
# x stands for inputs
X = np.array(([8,.8],[8,1],[8,.4],[4,.6],[5,.6],[2,.2],[4,1]), dtype=float)
y = np.array(([9],[9],[6],[5],[4],[2],[8]), dtype=float)
# print X
# print y

# Step 2
# Scale data (so neural net can compare apples to apples)
# i.e. divide by the maximum value for each dimension
X = X/np.amax(X, axis=0)
y = y/10 # Max mood is 10

print X
print y

# Step 3 - Build neural net
# i.e 2 neurons w/ 3 hidden layers
# (not sure where the 3 hidden layers come from)

class Neural_Network(object):
	def __init__(self):
		#Define HyperParameters
		#(constants that establish the structure and behavior of our network)
		self.inputLayerSize = 2
		self.outputLayerSize = 1
		self.hiddenLayerSize = 3

		#Weights (Parameters)
		# first set of synapses (weights)
		self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
		# second set of synapses (weights)
		self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)

	# Use matrices to pass multiple inputs at once
	def forward(self, X):
		#Propagate inputs through network
		# Use .dot for matrix multiplication
		self.z2 = np.dot(X, self.W1) #multiply X (original values) times W1
		self.a2 = self.sigmoid(self.z2) #apply sigmoid activation function to hidden neurons
		self.z3 = np.dot(self.a2, self.W2) #multiply a2 (hidden values) times W2
		yHat = self.sigmoid(self.z3)
		return yHat

	def sigmoid(self, z):
		# Apply sigmoid activation function to scalar, vector, or 
		return 1/(1+np.exp(-z))
	
	# Add a method for derivative of sigmoid function
	def sigmoidPrime(self, z):
		# Derivative of Sigmoid Function
		return np.exp(-z)/((1+np.exp(-z))**2)

	def costFunction(self, X, y):
	        #Compute cost for given X,y, use weights already stored in class.
	        self.yHat = self.forward(X)
	        J = 0.5*sum((y-self.yHat)**2)
	        return J

	def costFunctionPrime(self, X, y):
		#Compute derivative with respect to W1 and W2
		self.yHat = self.forward(X)

		# Find derivative of W2
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)

		# Find derivative of W1
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)

		return dJdW1, dJdW2

	# Helper functions
	def getParams(self):
		#Get W1 and W2 rolled into vector:
		params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
		return params

	def setParams(self, params):
		#Set W1 and W2 using single parameter vector:
		W1_start = 0
		W1_end = self.hiddenLayerSize*self.inputLayerSize
		self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
		W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
		self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

	def computeGradients(self, X, y):
		dJdW1, dJdW2 = self.costFunctionPrime(X, y)
		return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N
        
    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))   
        
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad
        
    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []
        
        params0 = self.N.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)
            
            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0
            
        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad 

# Create instance of Neural Network class
NN = Neural_Network()

# Call the 'forward' method to forward propogate, passing in the training dataset X
yHat = NN.forward(X) 

print yHat
print y

cost1 = NN.costFunction(X,y)
dJdW1, dJdW2 = NN.costFunctionPrime(X,y)

print dJdW1
print dJdW2

#Numerical Gradient Checking
#(Helps verify that gradients are correct)

def f(x):
	return x**2

epsilon = 1e-4
x = 1.5

numericGradient = (f(x+epsilon) - f(x-epsilon))/(2*epsilon)

print numericGradient
print 2*x

numgrad = computeNumericalGradient(NN, X, y)
grad = NN.computeGradients(X, y)

# compare the vectors
# using the Linear Algebra (linalg) 'norm' function
print np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad)

# Here's the meat! This is where we call the training function
T = trainer(NN)
T.train(X, y)

print NN.forward(X)
print y


# Avoiding overfitting

# Training Data
# trainX = 
# trainY = 

# # Testing Data
# textX = 
# testY = 

# # Normalize
# trainX = trainX/np.amax(trainX, axis=0)
# trainY = trainY/10

# test X = testX/np.amax(trainX, axis=0) #maximum from training data
# trainY = testY/10




