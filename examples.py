import numpy as np
from devices import pasqalobj
from operations import noise_evolve, states
import matplotlib.pyplot as plt

########################################

# Define some functions for the time-dependent parameters
def f_delta(t,args=2*np.pi/7):
    return np.sin(t)

def f_omega(t,args=0):
    return np.cos(t)

############################################
'''
if __name__ == "__main__":
    N = 8
    coords = 100 * np.random.rand(N,2) # Set atom coordinates at random
    noise_evolve.local_model(coords, 6, 'ising', f_delta, f_omega)
    plt.show()
'''

import sequences, bursts

if __name__ == "__main__":

    # Initial parameters
    my_rabi = "guess_rabi"
    my_detuning = "guess_detuning"
    my_interval = "guess_interval"
    InitialGuess = bursts.Parameters(my_rabi, my_detuning, my_interval)

    my_state = "ALL_UP"

    seq = sequences.QaoaSequence("shadoq2", Parameters=InitialGuess, layers=4, initial_state=my_state)
    print(seq.queue)
