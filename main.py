import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# Generate randon input data to train on
from tensorflow.contrib.distributions.python.ops.bijectors import inline
observations = 1000
xs = np.random.uniform(low=-10, high=10,size = (observations, 1))
zs = np.random.uniform(low=-10, high=10,size = (observations, 1))
#
# Combine x and z into an input matric to get 1000 by 2 input matrix
inputs = np.column_stack([xs,zs])
print (f'inputs shape: {inputs.shape}')
#
# Create targets that we will aim to
# In this cas, we have chosen targets = f(x,z) = 2*x - 3*z + 5 + noise .  The funtion is just an example
# 2 is weight1 (w1) and -3 is weight2 (w2) and 5 is the bias(b)
# noise is introduced to randomize the data a bit
noise =  np.random.uniform(low=-1, high=+1,size = (observations, 1))
print (f'noise shape: {noise.shape}')
#
# Construct the target
#targets = 2*xs - 3*zs + 5 + noise
targets = 13*xs - 7*zs - 12 + noise
print (f'targets shape: {targets.shape}')
#
# plot 3d plot to visuslize the data
# targets = targets.reshape(observations,)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.plot(xs,zs,targets)
# ax.set_xlabel('xs')
# ax.set_ylabel('zs')
# ax.set_zlabel('Targets')
# ax.view_init(azim=100)
# plt.show()
#
targets = targets.reshape(observations,1)
#
# y - xw + b
# y is the target, vary w and b
# x is the input
init_range = 0.1
weights = np.random.uniform(low=-init_range, high=+init_range,size = (2, 1))
biases =  np.random.uniform(low=-init_range, high=+init_range,size = 1)
#
# Assign a learning rate
learning_rate= .01
#
# Train the model
# ### calculate output
# ### compare outputs to targets through the loss
# ### print loss
# ### adjust weights and biases
# ### repeat
print(f'weights shape {weights.shape}')
print(f'biases shape {biases.shape}')
for i in range (5000):
    outputs = np.dot(inputs, weights) + biases
    deltas = outputs - targets
    #
    #L2 - norm loss formula is sum(yi-ti)**2
    # averae loss per observation
    loss = np.sum(deltas**2)/2/observations
    print(f'iter{i} --- loss = {loss}')
    deltas_scaled = deltas/observations
    # wi+1 = wi - eta * sum(xi * deltas)
    weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
    # weights is 2 X 1
    # learnig_rate is scalar
    # np.dot(inputs.T,deltas_scaled) is 2 X 1
    biases = biases - learning_rate * np.sum(deltas_scaled)

# targets = f(x,z) = 2*x - 3*z + 5 + noise
# compare weights to the coeff and bias in targets
print(weights,biases)
# plt.plot(outputs, targets)
# plt.xlabel('Outputs')
# plt.ylabel('Targets')
# plt.show()

