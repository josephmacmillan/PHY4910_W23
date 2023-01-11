import numpy as np
    
def ode_euler(x_start, x_end, h, y0, z0, f, g):
    """
    Solves a coupled pair of ODEs using euler
     
    Takes as arguments:
      x_start - starting point for independent coordinate
      x_end - ending point
      h - difference between x_i and x_i+1 (i.e., delat x)
      y0 - initial value for first variable y(x_start)
      z_0 - initial value for second variable z(x_start)
      f - function for derivative of first variable (i.e., f = dy/dx)
      g - function for deriviative of second variable (i.e., g = dz/dx)
        
      returns three arrays, x[0,N-1], y[0,N-1], and z[0,N-1].
    """
    
    x = np.arange(x_start, x_end, h)
    N = len(x)
    y = np.zeros(N)
    y[0] = y0
    z = np.zeros(N)
    z[0] = z0
    
    for i in range(0, N-1):
        k1 = h * f(x[i], y[i], z[i])
        l1 = h * g(x[i], y[i], z[i])
            
        y[i+1] = y[i] + k1
        z[i+1] = z[i] + l1
                
    return x, y, z
    


def ode_rk4(x_start, x_end, h, y0, z0, f, g):
    """ 
    Solves a coupled pair of ODEs using runge kutta.
     
    Takes as arguments:
      x_start - starting point for independent coordinate
      x_end - ending point
      h - difference between x_i and x_i+1 (i.e., delat x)
      y0 - initial value for first variable y(x_start)
      z_0 - initial value for second variable z(x_start)
      f - function for derivative of first variable (i.e., f = dy/dx)
      g - function for deriviative of second variable (i.e., g = dz/dx)
      stop - a function for a stopping criteria.  Function must take three number (xi, yi, zi).
      data - any data that needs to be passed to the stop function in addition to xi, yi, zi.
      
      returns three arrays, x[0,N-1], y[0,N-1], and z[0,N-1].
    """
    
    x = np.arange(x_start, x_end, h)
    N = len(x)
    y= np.zeros(N)
    y[0] = y0
    z = np.zeros(N)
    z[0] = z0
    
    for i in range(0, N-1):
        k1 = h * f(x[i], y[i], z[i])
        l1 = h * g(x[i], y[i], z[i])
        k2 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k1, z[i] + 0.5 * l1)
        l2 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * k1, z[i] + 0.5 * l1)
        k3 = h * f(x[i] + 0.5 * h, y[i] + 0.5 * k2, z[i] + 0.5 * l2)
        l3 = h * g(x[i] + 0.5 * h, y[i] + 0.5 * k2, z[i] + 0.5 * l2)
        k4 = h * f(x[i] + h, y[i] + k3, z[i] + l3)
        l4 = h * g(x[i] + h, y[i] + k3, z[i] + l3)
    
        y[i+1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        z[i+1] = z[i] + (l1 + 2.0 * l2 + 2.0 * l3 + l4) / 6.0
        
    return x, y, z
