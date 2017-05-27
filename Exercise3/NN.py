import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from numpy.matlib import repmat as y

# initialize the weights between units with the standard normal distribution
# (mean:0 variance:1) and the bias weights with 0 for given network architecture
def initialize(input_size, hidden_size, output_size):
    ## YOUR CODE HERE ##
    W1 = 0
    b1 = 0
    W2 = 0
    b2 = 0
    # InputToHidden        HiddenToOutput
    # 400X30         -->       30X5
    W1 = np.random.randn(input_size, hidden_size) # weight of input to hidden layer should be 400X30
    W2 = np.random.randn(hidden_size, output_size) # weight of hidden to output layers should be 30X5
    b1 = np.zeros((1,hidden_size)) # Bias of hidden
    b2 = np.zeros((1,output_size)) # Bias of output
    ####################
    return W1, b1, W2, b2

# calculate the output of the activation function for the accummulated signal x
def calcActivation(x):
    ## YOUR CODE HERE ##
    z = 0
    z = 1.0 / (1 + (np.e ** (-x)))
    ####################
    return z

# propagate the input and calculate the output at each layer
def forwardPropagate(X, W1, b1, W2, b2):
    ## YOUR CODE HERE ##
    Z1 = 0
    Z2 = 0

    Z1 = calcActivation(np.dot(X, W1) + b1)
    Z2 = calcActivation(np.dot(Z1, W2) + b2)
    ####################
    return Z1, Z2

def calcCost(Z2,y):
    ## YOUR CODE HERE ##
    cost = 0
    ms = len(y)
    for m in range(ms):
       if y[m] == 0:
           cost += -(np.log(1 - Z2[0][m]) * (1 - y[m]))
       else:
           cost += -(np.log(Z2[0][m]) * y[m])
    ####################
    return cost

# propagate the error and calculate the errors at the output and the hidden layer
def backPropagate(Z1, Z2, y, W2, b2):
    ## YOUR CODE HERE ##
    E2 = 0
    E1 = 0
    Eb1 = 0
    
    E2 = Z2 - y # Find error of output
    E1 = np.dot(E2, np.transpose(W2)) # Find error of hidden layer
    Eb1 = np.dot(b2, np.transpose(E2)) # Find error on bias
    ####################
    return E2, E1, Eb1

# calculate the gradients for weights between units and the bias weights
def calcGrads(X, Z1, Z2, E1, E2, Eb1):
    ## YOUR CODE HERE ##
    d_W1 = 0
    d_b1 = 0
    d_W2 = 0
    d_b2 = 0

    d_b2 = np.dot(Eb1, E2) # bias of output
    d_b1 = E1 # bias is error itself 
    d_W2 = np.dot(np.transpose(E2), Z1)
    d_W1 = np.dot(np.transpose(X),E1)
   
    ####################
    return d_W1, d_W2, d_b1, d_b2

# update the weights between units and the bias weights using a learning rate of alpha
def updateWeights(W1, b1, W2, b2, alpha, d_W1, d_W2, d_b1, d_b2): # Update values with learning rate
    ## YOUR CODE HERE ##
    W1 = W1 - alpha * d_W1 
    W2 = W2 - alpha * d_W2.T  
    b1 = b1 - alpha * d_b1
    b2 = b2 - alpha * d_b2
    ####################
    return W1, b1, W2, b2

np.random.seed(62)
X = np.random.randn(400,1).T
y = [0, 0, 1, 0, 0]
input_size = 400
hidden_size = 30
output_size = 5
alpha = 0.001
num_iter = 100
OUT = []
COST = []

# implement the iterations for neural network training, append the output and
# the cost at each iteration to their corresponding lists
W1, b1, W2, b2 = initialize(input_size, hidden_size, output_size)
for i in range(num_iter):
    ## YOUR CODE HERE ##
    Z1, Z2 = forwardPropagate(X, W1, b1, W2, b2) # First FP to get output
    OUT.append(Z2)
    cost = calcCost(Z2,y) # Find cost
    COST.append(cost)
    E2, E1, Eb1 = backPropagate(Z1, Z2, y, W2, b2) # Find errors
    d_W1, d_W2, d_b1, d_b2 = calcGrads(X, Z1, Z2, E1, E2, Eb1) # Calculate new values
    W1, b1, W2, b2 = updateWeights(W1, b1, W2, b2, alpha, d_W1, d_W2, d_b1, d_b2) # Update according to new values
    ####################

# plotting part is already implemented for you
cm_subsection = np.linspace(0, 1, num_iter)
colors = [ cm.cool(x) for x in cm_subsection ]
plt.figure()
plt.hold(True)
for i in range(num_iter): plt.plot(OUT[i][0], color=colors[i])
plt.figure()
plt.hold(True)
plt.plot(COST)
plt.show()
