import numpy as np
from operations import noise_evolve, states
import matplotlib.pyplot as plt

import sequences, pulses

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

def main():
    # Initial parameters
    depth = 5

    # Define parameters:
    params = {
                "rabi" : [f"rabi{k}" for k in range(depth)],
                "detuning" : [f"detuning{k}" for k in range(depth)],
                "intervals" : [ 0.1*k for k in range(depth)]
                }
    # can add also identifiers for a particular type of function (square, sawtooth, etc).
    # the associated parameters should still remain as numbers

    seq = ParamSequence(device="my_device", param_set=params)
    print(seq.queue)


if __name__ == "__main__":
    main()
