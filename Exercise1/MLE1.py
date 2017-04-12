import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Part 1: Load the data and separate it into three arrays; one for each class
data = np.loadtxt('MLE1_iris.data')
data0 = data[0:50,:]
data1 = data[50:100,:]
data2 = data[100:150,:]
# ^^ YOUR CODE HERE ^^

# Part 2: Plot each typr of data for all classes in 1D (with shifts of 0.1 for better visualization)
fig = plt.figure()
plt.plot(data0[:,0], np.ones(len(data0[:,0]))*0.0, '+r', label='Data 0 Class 0')
plt.plot(data1[:,0], np.ones(len(data1[:,0]))*0.1, '+g', label='Data 0 Class 1')
plt.plot(data2[:,0], np.ones(len(data2[:,0]))*0.2, '+b', label='Data 0 Class 2')

plt.plot(data0[:,1], np.ones(len(data0[:,1]))*1.0, 'xr', label='Data 1 Class 0')
plt.plot(data1[:,1], np.ones(len(data1[:,1]))*1.1, 'xg', label='Data 1 Class 1')
plt.plot(data2[:,1], np.ones(len(data2[:,1]))*1.2, 'xb', label='Data 1 Class 2')

plt.plot(data0[:,2], np.ones(len(data0[:,2]))*2.0, '.r', label='Data 2 Class 0')
plt.plot(data1[:,2], np.ones(len(data1[:,2]))*2.1, '.g', label='Data 2 Class 1')
plt.plot(data2[:,2], np.ones(len(data2[:,2]))*2.2, '.b', label='Data 2 Class 2')

plt.plot(data0[:,3], np.ones(len(data0[:,3]))*3.0, '1r', label='Data 3 Class 0')
plt.plot(data1[:,3], np.ones(len(data1[:,3]))*3.1, '1g', label='Data 3 Class 1')
plt.plot(data2[:,3], np.ones(len(data2[:,3]))*3.2, '1b', label='Data 3 Class 2')

plt.legend(fontsize=9, loc=3)

# Part 3: Examining the plots above select two of the data types and plot them in 2D - one data type for each axis. Let's say you chose ath and bth columns as your data.
# This means you have to plot dataN[:,a] vs dataN[:,b] for N=0,1,2.
# ^^ YOUR CODE HERE ^^


fgr = plt.figure()
plt.plot(data0[:,2],data0[:,3], 'xr')
plt.plot(data1[:,2],data1[:,3], '+y')
plt.plot(data2[:,2],data2[:,3], '1b')


# Part 4: Using the two datatype you have chosen, extract the 2D Gaussian (Normal) distribution parameters. Numpy functions are called here to be used ONLY for validation of your results.
#mx0 = np.mean(data0[:,2])
#my0 = np.mean(data0[:,3])
#cov0 = np.cov(data0[:,2:4].T)
#mx1 = np.mean(data1[:,2])
#my1 = np.mean(data1[:,3])
#cov1 = np.cov(data1[:,2:4].T)
#mx2 = np.mean(data2[:,2])
#my2 = np.mean(data2[:,3])
#cov2 = np.cov(data2[:,2:4].T)
# ^^ YOUR CODE HERE ^^

data02 = data0[:,2]
data03 = data0[:,3]
data12 = data1[:,2]
data13 = data1[:,3]
data22 = data2[:,2]
data23 = data2[:,3]

def mean(list):
    return np.sum(list) / len(list)

def cov(lst,lst1):
    ln = len(lst)
    sm = 0
    for (i,j) in zip(lst,lst1):
        sm = (i - mean(lst)) * (j - mean(lst1)) + sm
    return sm / (ln - 1)

mx0 = mean(data02)
my0 = mean(data03)
mx1 = mean(data12)
my1 = mean(data13)
mx2 = mean(data22)
my2 = mean(data23)
cov0 = np.array([[cov(data02,data02),cov(data02,data03)],[cov(data03,data02),cov(data03,data03)]])
cov1 = np.array([[cov(data12,data12),cov(data12,data13)],[cov(data13,data12),cov(data13,data13)]])
cov2 = np.array([[cov(data22,data22),cov(data22,data23)],[cov(data23,data22),cov(data23,data23)]])

# Part 5: Plot the Gaussian surfaces for each class.
## First, we generate the grid to compute the Gaussian function on.
vals = np.linspace(np.min(data),np.max(data) , 500)
x,y = np.meshgrid(vals, vals)

def gaussian_2d(x,y,mx,my,cov):
    ''' x and y are the 2D coordinates to calculate the function value
        mx and my are the mean parameters in x and y axes
        cov is the 2x2 variance-covariance matrix'''
    ret = 0
    # ^^ YOUR CODE HERE ^^
    p = 0
    p = cov[0][1] / (np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))

    e1 = 1 / (2 * np.pi * np.sqrt(cov[0][0])\
            * np.sqrt(cov[1][1])\
            * np.sqrt(1-pow(p,2)))
    e2 = -1 / (2 * (1-pow(p,2))) * ((((x-mx)**2)\
            / (np.sqrt(cov[0][0])**2))\
            + (((y-my)**2)/(np.sqrt(cov[1][1]))**2)\
            - ((2*p*(x-mx)*(y-my))\
            / (np.sqrt(cov[0][0])*np.sqrt(cov[1][1]))))
    ret = e1 * pow(np.e,e2)
    return ret

## Finally, we compute the Gaussian function outputs for each entry in our mesh and plot the surface for each class.
z0 = gaussian_2d(x, y, mx0, my0, cov0)
z1 = gaussian_2d(x, y, mx1, my1, cov1)
z2 = gaussian_2d(x, y, mx2, my2, cov2)
fig0 = plt.figure()
ax0 = fig0.add_subplot(111, projection='3d')
ax0.plot_surface(x, y, z0, cmap=cm.jet, linewidth=0, antialiased=False)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot_surface(x, y, z1, cmap=cm.jet, linewidth=0, antialiased=False)
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot_surface(x, y, z2, cmap=cm.jet, linewidth=0, antialiased=False)

plt.show()
# Part 6: Classify each sample in the dataset based on your findings and assign a class label. Explain your reasoning behind your implementation with few sentences

lbl = []
for d in data:
    label = 0
    # ^^ YOUR CODE HERE ^^
    d0 = [0,gaussian_2d(d[2],d[3],mx0,my0,cov0)]
    d1 = [1,gaussian_2d(d[2],d[3],mx1,my1,cov1)]
    d2 = [2,gaussian_2d(d[2],d[3],mx2,my2,cov2)]
    label = max(d0,d1,d2,key=lambda s: s[1])[0]
    lbl.append(label)

# Reasoning
# Because of maximum likelihood, to find class of our dataset, we are trying 3 gaussian and taking maximum out of them.

# Part 7: Calculate the success rate - the percentage of correctly classified samples
success_rate = 0
# ^^ YOUR CODE HERE ^^
for d,l in zip(data,lbl):
    if int(d[4]) is int(l):
        success_rate +=1
success_rate=(success_rate/150.0) * 100.0

print 'Success rate is %4.2f %%' %success_rate

# Part 8: Repeat the same process for non-overlapping training and test sets.
data_test = np.vstack((data[0:25],data[50:75],data[100:125]))
data_train = np.vstack((data[25:50],data[75:100],data[125:150]))
data_test0 = data_test[data_test[:,4]==0]
data_test1 = data_test[data_test[:,4]==1]
data_test2 = data_test[data_test[:,4]==2]
data_train0 = data_train[data_train[:,4]==0]
data_train1 = data_train[data_train[:,4]==1]
data_train2 = data_train[data_train[:,4]==2]

data_train02 = data_train0[:,2]
data_train03 = data_train0[:,3]
data_train12 = data_train1[:,2]
data_train13 = data_train1[:,3]
data_train22 = data_train2[:,2]
data_train23 = data_train2[:,3]

mxt0 = mean(data_train02)
myt0 = mean(data_train03)
mxt1 = mean(data_train12)
myt1 = mean(data_train13)
mxt2 = mean(data_train22)
myt2 = mean(data_train23)
covt0 = np.array([[cov(data_train02,data_train02),cov(data_train02,data_train03)],[cov(data_train03,data_train02),cov(data_train03,data_train03)]])
covt1 = np.array([[cov(data_train12,data_train12),cov(data_train12,data_train13)],[cov(data_train13,data_train12),cov(data_train13,data_train13)]])
covt2 = np.array([[cov(data_train22,data_train22),cov(data_train22,data_train23)],[cov(data_train23,data_train22),cov(data_train23,data_train23)]])


lblt = []
for d in data_test:
    label = 0
    # ^^ YOUR CODE HERE ^^
    d0 = [0,gaussian_2d(d[2],d[3],mxt0,myt0,covt0)]
    d1 = [1,gaussian_2d(d[2],d[3],mxt1,myt1,covt1)]
    d2 = [2,gaussian_2d(d[2],d[3],mxt2,myt2,covt2)]
    label = max(d0,d1,d2,key=lambda s: s[1])[0]
    lblt.append(label)

success_ratet = 0
# ^^ YOUR CODE HERE ^^
for d,l in zip(data_test,lblt):
    if int(d[4]) is int(l):
        success_ratet +=1
success_ratet=(success_ratet/75.0) * 100.0

print 'Success rate is %4.2f %%' %success_ratet


