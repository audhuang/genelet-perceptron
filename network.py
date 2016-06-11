from __future__ import print_function
from __future__ import division
import random
import numpy as np 
from numpy.linalg import inv


def inv_perceptron(weights, thresh, states, num):
	while True: 
		num_wrong = 0
		
		for b in range(num): 
			p = states[:,b]
			
			for j in range(num): 
				if np.dot(weights[:, j], p) >= thresh[j] and p[j] == 1: 
					weights[:, j] = weights[:, j] - p
					thresh[j] += 1
					num_wrong += 1
				elif np.dot(weights[:, j], p) < thresh[j] and p[j] == 0: 
					weights[:, j] = weights[:, j] + p
					thresh[j] -= 1
					num_wrong += 1
		
		print("number of incorrectly classified points: ", num_wrong)
		if num_wrong == 0: 
			break 
	
	return weights

def perceptron(weights, thresh, states, num):
	while True: 
		num_wrong = 0
		
		for b in range(num): 
			p = states[:,b]
			
			for j in range(num): 
				if np.dot(weights[:, j], p) < thresh[j] and p[j] == 1: 
					weights[:, j] = weights[:, j] + p
					thresh[j] -= 1
					num_wrong += 1
				elif np.dot(weights[:, j], p) >= thresh[j] and p[j] == 0: 
					weights[:, j] = weights[:, j] - p
					thresh[j] += 1
					num_wrong += 1
		
		print("number of incorrectly classified points: ", num_wrong)
		if num_wrong == 0: 
			break 
	
	return weights

if __name__ == '__main__':
 	n_ij = 4
 	n_w = 4 

 	weights = np.zeros(4, 4)
 	thresh = np.zeros(1, 4)[0]
 	states = np.asarray([[1, 0, 0 , 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

 	weights = perceptron(weights, thresh, states, n_ij)
 	print("final weights: \n", weights)
 	min_weight = abs(np.amin(weights))
 	print("lowest negative weight: ", -1 * min_weight)
 	print("adjusted weights: \n", np.add(weights, min_weight))


 	

