import numpy as np
#from operations import noise_evolve, states
import matplotlib.pyplot as plt

import devices, sequences, pulses

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
    N = 7
    np.random.seed(3)
    atoms = np.random.randint(0,40,(N,2))

    # Depth of the circuit
    depth = 5

    # Define parameters:
    params = {
                "rabi" : [np.random.rand() for k in range(depth)],
                "detuning" : [np.random.rand() for k in range(depth)],
                "intervals" : [ 0.1*k for k in range(depth)]
                }
    # can add also identifiers for a particular type of function (square, sawtooth, etc).
    # the associated parameters should still remain as numbers

    # Define Device
    my_device =devices.Device1(atoms)

    seq = sequences.ParamSequence(device="my_device", param_set=params)

    param_set = seq.get_parameter_set()
    print(param_set)

if __name__ == "__main__":
    main()
