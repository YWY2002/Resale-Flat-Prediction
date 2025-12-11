import numpy as np
import functions as func

testing_w = np.array([2, 1])

example_data = [func.generateExamples(testing_w) for i in range(500)]
# x, y = [func.generateExamples(testing_w) for i in range(500)]
# print(example_data)
func.batchGradientDescent(func.mseLoss, func.lossGradient, len(testing_w), example_data, func.linearPhi)