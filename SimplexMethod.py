#Custom Built Simplex Method
import numpy as np

#Assume the constraints are of the form Ax=b, for x>= 0
def simplexMethod(A,b,c):
	#We will be using the Big M Method
	#First, we must transform the constrains Ax=b so that b >= 0 and we have enough
	#slack variables and artificial variables, allowing our first feasible solution to be
	#(0,0,0)  for the x_i in the objective function
	for i in range(len(b)):
		if float(b[i]) < 0:
			b[i] = -1*b[i]
			A[i] = -1*A[i]
	
	length = A.shape[1] #This is saved so that we know what the original length of the solution should be before we add
						#slack variables
	
	#Next, add slacks and articifial variables. I will do this to every constraint to make it simple
	#We might have extra slack varaibles than necessary. Thats okay, all it does is increase computation time,
	#but for the sake of simplicity, do not worry about it
	A = np.hstack([A, np.eye(A.shape[0])])
	
	#Add to the costs vector c to include costs for the artificial variables
	#Remember the new function is c dot x + M\sum y_i, where y_i are the artificial variables, and M >> 0.
	#Choose M = 1,000,000
	M = 10000
	c = np.hstack([c,np.full(A.shape[0],M)])
	
	
	#The LP is transformed. Now we can begin the Simplex Method
	
	x = np.hstack([np.zeros(A.shape[1]-len(b)),b]) #Initial basic feasible solution
	
	#Basis. As is the case for any LP after the constraints are transformed, the intitial basis will be
	 #the indices of the last len(b) variables 
	basis = list(range(0,A.shape[1]))[A.shape[1]-len(b):]
	
	
	#Calculate reduced costs
	
	invB_times_A = np.linalg.solve(A[:,basis], A)
	
	reduced_costs = np.zeros(A.shape[1])
	for i in range(A.shape[1]):
		if i not in basis:
			#Variable index is in the basis. So variable is basic
			reduced_costs[i] = c[i] - np.transpose(c[basis]) @ invB_times_A[:,i]
	
	
	
	#Check if all reduced costs are non positive
	invB_times_A = np.linalg.solve(A[:,basis], A)
	while np.any(reduced_costs < -0.000001): #when working wtih big matrices and big numbers, float point accuracy may mis calculate zeros
		#There exists negative reduced costs. Our solution is not optimal yet
		#Find the smallest index j such that that reduced cost is negative. This index will enter the basis.
		j = 0
		while reduced_costs[j] >= -0.000001: j = j + 1
		
		#Compute   d and u = B^(-1)*A_j = -d_B
		d = np.zeros(A.shape[1])
		d[j] = 1
		d[basis] = -1*invB_times_A[:,j]
		u = -1*d[basis]
		if np.all(u <= 0) :
			#The LP is unbounded. Terminate algorithm
			return "Unbounded"
		else:
			
			#Perform ratio test
			ratios = []
			viableRatios_Bases = []
			for i in basis:
				if d[i] < 0:
					ratios.append(-1*x[i] / d[i])
					viableRatios_Bases.append(i)
					 
			
			theta = min(ratios)
			l = viableRatios_Bases[ratios.index(theta)] #this basis index will be replaced by j
			
		
		#Replace index l for index j in the new basis
		for i in range(len(basis)):
			if basis[i] == l:
				basis[i] = j
		#new solution
		x = x + theta*d
		
		
		
		#Compute new reduced costs
		invB_times_A = np.linalg.solve(A[:,basis], A)
		reduced_costs = np.zeros(A.shape[1])
		for i in range(A.shape[1]):
			if i not in basis:
				#Variable index is in the basis. So variable is basic
				reduced_costs[i] = c[i] - np.transpose(c[basis]) @ invB_times_A[:,i]
		
	#Now that we have found an optimal solution to the transformed LP, we need to check if there are any 
	#artificial variables in the basis now. If so, then the original LP is unbounded. we can do this by considering the 
	#starting basis and current basis, converting them into sets, and checking their intersectio
	
	artificialIndices = list(range(0,A.shape[1]))[A.shape[1]-len(b):]
	if set(artificialIndices).intersection(set(basis)) != set():
		#There is an artificial index in the basis for the optimal solution. The original LP is thus infeasible
		return "infeasible"
	else:
		return x[:length]
