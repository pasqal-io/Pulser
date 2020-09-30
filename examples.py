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
    my_rabi = "rabi_guess"
    my_detuning = "guess_detuning"
    my_interval = "guess_interval"
    InitialGuess = pulses.Parameters(my_rabi, my_detuning, my_interval)

    # cost_fct =
    my_state = "ALL_UP"
    my_device = "devices.Device1(...)"

    seq = sequences.ParamSequence(device=my_device, Parameters=InitialGuess, initial_state=my_state)
    print(seq.queue)


if __name__ == "__main__":
    main()
