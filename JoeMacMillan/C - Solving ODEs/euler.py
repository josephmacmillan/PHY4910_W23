import numpy as np
from phy4910 import ode_euler

#
# Uses Euler to solve the Lane Emden equation
#

n = 1

def f(x, y, z):
	return z
	
def g(x, y, z):
	return -2/x*z - np.power(y, n)

eta, rho, sigma = ode_euler(0.001, 5, 0.01, 1, 0, f, g)

# what's the actual solution? 
rho_actual = np.sin(eta) / eta

# print out the data so we can plot it later
np.savetxt("rk4_data.txt", np.column_stack((eta, rho, rho_actual)))

