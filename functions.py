import numpy as np
import math

def generateExamples(weight_vector):
    x = np.random.randn()
    y = weight_vector[0] * x + weight_vector[1] + np.random.rand()   # Linear
    # print(f"Example: {x}, {y}")
    return x, y

def linearPhi(x):
    return np.array([x, 1])
    # x = f[0]
    # c = f[1]
    # return np.array([x, c])

def quadraticPhi(x):
    return np.array([x**2, x, 1])

def initWeightVector(n):
    return np.zeros(n)

def mseLoss(weight, training_data, phi):
    # TODO Return MSE
    mse = 1 / len(training_data) * sum((weight.dot(phi(x)) - y) ** 2 for x, y in training_data)
    return mse 

def lossGradient(weight, training_data, phi):
    # TODO Return loss gradient with respect to weight
    grad = 1 / len(training_data) * sum(2 * (weight.dot(phi(x)) - y) * phi(x) for x, y in training_data)
    return grad

def batchGradientDescent(loss, lossGradient, feature_dim, training_data, phi):
    w = initWeightVector(feature_dim)
    max_epoch = 500

    for i in range(max_epoch):
        cost = loss(w, training_data, phi)
        gradient = lossGradient(w, training_data, phi)
        step = 0.1
        # step = 1 / math.sqrt(i + 1)
        w = w - step * gradient
        print(f'epoch: {i} Cost: {cost:.8f} Loss_gradient: {gradient} Weight: {w}')