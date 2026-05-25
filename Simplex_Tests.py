#Example using the Simplex and Tableau Simplex Methods for solving linear programs. 
#Assume the LP is in standard form with the variables x>=0

import numpy as np
import SimplexMethod
import SimplexTableau

A = np.array([[1, 2, 2, 1, 0, 0],
			  [2, 1, 2, 0, 1, 0],
			  [2, 2, 1, 0, 0, 1]])

b = np.array([20,20,20])

c = np.array([-10,-12,-12,0,0,0])

simplex_solution = SimplexMethod.simplexMethod(A,b,c)
tableau_solution = SimplexTableau.simplexTableau(A,b,c)
print(simplex_solution)
print(tableau_solution)