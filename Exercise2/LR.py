# Cem GURESCI 200201027

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Part 1: Load the data from file
'''YOUR CODE HERE'''
data = np.loadtxt("LR_population_vs_profit.data")
# Part 2: Plot the data in 2D as population (X) vs. profit (y)
plt.figure(1)
'''YOUR CODE HERE'''
plt.xlabel('City Population in 10.000s')
plt.ylabel('Profit in $10.000s')

dataC = data[0:96,0]
dataP = data[0:96,1]

plt.plot(dataC,dataP,'.r')
#plt.show()

# If you like, you can see this plot by calling plt.show() at this line. However, in order to be able to plot the estimated lines on top of the same figure in Parts 7 and 9 (that's why an ID (1) is given), it is not be plotted here.)

# Part 3: Implement the gradient step calculation for theta
def computeStep(X,y,theta):
    '''YOUR CODE HERE'''
    m = float(len(X))
    sum0 = sum1 = 0
    for i in range(len(X)):
        x0 = X[i,0]
        x1 = X[i,1]
        hX = theta[0]*x0 + theta[1]*x1
        sum0 += hX - y[i]
        sum1 += (hX - y[i]) * x1
    temp0 = (1/m) * sum0
    temp1 = (1/m) * sum1
    
    theta1 = np.array([temp0,temp1])
    return theta1


# Part 4: Implement the cost function calculation
def computeCost(X,y,theta):
    '''YOUR CODE HERE'''
    m = float(len(X))
    tmpSum = 0
    for i in range(len(X)):
        x0 = X[i,0]
        x1 = X[i,1]
        hX = theta[0]*x0 + theta[1]*x1
        tmpSum += (hX - y[i]) ** 2 
    cost = tmpSum / (2*m)
    return cost

# Part 5: Prepare the data so that the input X has two columns: first a column of ones to accomodate theta0 and then a column of city population data
'''YOUR CODE HERE'''

ones = np.ones((len(dataC),1))
population = dataC
population = np.reshape(population, (len(population), 1))
X = np.column_stack((ones,population))

# Part 6: Apply linear regression with gradient descent
num_iter = 1500
alpha_line = [[0.1, '-b'], [0.03, '-r'], [0.01, '-g'], [0.003, ':b'], [0.001, ':r'], [0.0003, ':g']]
theta = np.array([0,0])

'''YOUR CODE HERE'''

y = dataP
init_cost = computeCost(X,y,theta)
print 'The initial cost is %f.' % init_cost

plt.figure()
plt.ylim(0,100)
plt.xlim(0,10)
final_theta = []
for alpha, line in alpha_line:
    J_history = []
    theta = np.array([0,0],dtype=float)
    for i in range(num_iter):
        '''YOUR CODE HERE'''
        stepTmp = computeStep(X,y,theta)
        theta[0] = theta[0] - alpha * stepTmp[0]
        theta[1] = theta[1] - alpha * stepTmp[1]
        J_history.append(computeCost(X,y,theta))

    plt.plot(J_history, line, linewidth=3, label='alpha:%5.4f'%alpha)
    final_theta.append(theta)
    print 'Final cost after %d iterations is %f.' %(num_iter, J_history[-1])
plt.legend(fontsize=12)
best_theta = final_theta[2]

# Part 7: Plot the resulting line and make predictions with the best performing theta 
plt.figure(1)
'''YOUR CODE HERE'''
plt.plot(X[:,1],np.dot(X,best_theta),'-', label='Linear regression with gradient descent')
#plt.show()

'''YOUR CODE HERE'''
y1 = 1*best_theta[0] + best_theta[1]*35
y2 = 1*best_theta[0] + best_theta[1]*70
print 'Estimated profit for a city of population 35000 is % 7.2f.' %y1
print 'Estimated profit for a city of population 70000 is % 7.2f.' %y2

# Part 8: Plot cost function as a 2D surface over theta0 and theta1 axes
grid_size = 200
theta0_vals = np.linspace(-10, 10, grid_size)
theta1_vals = np.linspace(-1, 4, grid_size)
theta0, theta1 = np.meshgrid(theta0_vals, theta1_vals)
cost_2d = np.zeros((grid_size,grid_size))
for t0 in range(grid_size):
    for t1 in range(grid_size):
        theta = [theta0[t0,t1],theta1[t0,t1]]
        cost_2d[t0,t1] = computeCost(X,y,theta)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(theta0, theta1, cost_2d, cmap=cm.jet, linewidth=0, antialiased=False, alpha=0.5 )
ax.set_xlabel('Theta 0')
ax.set_ylabel('Theta 1')
ax.set_zlabel('Cost')
plt.figure()
plt.contour(theta0, theta1, cost_2d, 100)
plt.plot(best_theta[0], best_theta[1], 'xr')
plt.xlabel('Theta 0')
plt.ylabel('Theta 1')
#plt.show()

# Part 9: Calculate optimal theta values using normal equation and then compute the corresponding cost value
'''YOUR CODE HERE'''
theta_normal = np.linalg.pinv(X).dot(y)
cost_normal = computeCost(X,y,theta_normal)
print 'Theta parameters obtained by solving the normal equation are  %f and %f.' %(theta_normal[0],theta_normal[1])
print 'Final cost after solving the normal equation is  %f.' %cost_normal

plt.figure(1)
plt.plot(X[:,1],np.dot(X,theta_normal),'-r', label='Linear regression with normal equation')
plt.legend(fontsize=12)
plt.show()
