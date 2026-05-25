#Tableau Method.
import numpy as np

#Assume the LP is in standard form.

def simplexTableau(A,b,c):
	length = A.shape[1] #This is saved so that we know what the original length of the solution should be before we add
						#slack variables
	#The beginning of this code starts the same as the normal simpelx method. 
	
	#We will be using the Big M Method
	#First, we must transform the constrains Ax=b so that b >= 0 and we have enough
	#slack variables and artificial variables, allowing our first feasible solution to be
	#(0,0,0)  for the x_i in the objective function
	for i in range(len(b)):
		if float(b[i]) < 0:
			b[i] = -1*b[i]
			A[i] = -1*A[i]
	
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
	reduced_costs = np.zeros(A.shape[1])
	for i in range(A.shape[1]):
		if i not in basis:
			#Variable index is in the basis. So variable is basic
			reduced_costs[i] = c[i] - np.transpose(c[basis]) @ np.linalg.inv(A[:,basis]) @ A[:,i]	
	#Formulate the block matrix Z: 
		# c_B^T*B^-1)*b  reduced_costs
		# B^(-1)*b       B^(-1)*A
	Z1 = np.hstack([-1*np.transpose(c[basis]) @ np.linalg.inv(A[:,basis]) @ b, reduced_costs])
	Z2 = np.column_stack([np.linalg.inv(A[:,basis]) @ b, np.linalg.inv(A[:,basis]) @ A])
	Z = np.vstack([Z1, Z2])
	
	
	while any(Z[0,1:] < 0):
		#CHoose the smallest j to enter the basis where the corresponding reduced cost in Z is non-negative.
		j = 1
		while Z[0,j] >= 0: 
			j += 1
		
		u = Z[1:,j]
		if np.all(u <= 0):
			#optimal cost is -infinity and the LP is unbounded. Terminate the algorithm
			return "unbounded"
		else:
			ratios = []
			ratios_index = []
			for i in range(len(u)):
				if u[i] > 0:
					ratios.append(Z[i+1,0] / u[i])
					ratios_index.append(i)
					
			theta = min(ratios)
			l = ratios_index[ratios.index(theta)] #this basis will be replaced by j in the matrix Z
			
			#index l of u is row l+1 of the matrix Z. We want to use the number of the matrix, so we will be using l+1
			l += 1
			
			#In each row, add a constant multiple of the lth row so that u(l) becomes
			#1 and all other entries in the jth column become 0.
			
			Z[l,:] = Z[l,:] / Z[l,j]
			for i in range(l):
				Z[i,:] = Z[i,:] - (Z[i,j] * Z[l,:])
			for i in range(l+1,Z.shape[0]):
				Z[i,:] = Z[i,:] - (Z[i,j] * Z[l,:])
		
		
			#update our basis
			x[basis[l-1]] = 0
			basis[l-1] = j-1
			x[basis[l-1]] = Z[l,0]
			
			#update our solution x
			x = np.zeros(len(Z[0,1:]))
	
			k = 1
			for i in basis:
				x[i] = Z[k,0]
				k += 1
	
	#Now that we have found an optimal solution to the transformed LP, we need to check if there are any 
	#artificial variables in the basis now. If so, then the original LP is unbounded. we can do this by considering the 
	#starting basis and current basis, converting them into sets, and checking their intersectio
	
	artificialIndices = list(range(0,A.shape[1]))[A.shape[1]-len(b):]
	if set(artificialIndices).intersection(set(basis)) != set():
		#There is an artificial index in the basis for the optimal solution. The original LP is thus infeasible
		return "infeasible"
	else:
		return x[:length]