import numpy as np
from device import pasqalobj
from operations import noise_evolve, states
import matplotlib.pyplot as plt

########################################

# Define some functions for the time-dependent parameters
def f_delta(t,args=2*np.pi/7):
    return np.sin(t)

def f_omega(t,args=0):
    return np.cos(t)

############################################

if __name__ == "__main__":
    N = 8
    coords = 100 * np.random.rand(N,2) # Set atom coordinates at random
    noise_evolve.local_model(coords, 6, 'ising', f_delta, f_omega)
    plt.show()
