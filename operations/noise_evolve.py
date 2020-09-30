from operations import states as st
from hamiltonians import pasqalobj
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
# Set an observable:
def define_local_observable(site,N):
    """
    Measures magnetization at site `site` (TODO: Generalize this!)
    """
    op_list = [qeye(2) for _ in range(N)]
    op_list[site] = sigmaz()
    observable = tensor(op_list)

    return observable


def local_model(coords, site, model, delta, omega):
    """
    A simple model to test the pasqalobj class with both Ising and XY hamiltonians.

    Parameters
    ----------

    N : int
        number of sites of the system
    site : int
        site where we place our sigmaz observable

    Returns
    -------

    A plot of the coordinate distribution and the time evolution (with a dephasing
    noise model) of the sigmaZ observable in the position 'site'.
    """

    N = len(coords)

    # Initial state and time array
    psi_0 = st.build_init_state(N)
    times = np.linspace(0.0, 10.0, 200)

    # Plot atom configuration
    plot_graph = plt.figure(0)
    plt.plot(coords[:,0], coords[:,1], "o")

    # Define observable
    observable = define_local_observable(site,N)

    plot_evol = plt.figure(1)

    if model == 'ising':
        ising = pasqalobj.IsingPasqalObj(coords, delta, omega, noise_model='dephasing')
        H0, HX, HZ = ising.build_hamiltonian(coords, C6 = 30)
        H = [H0, [HX, ising.omega], [HZ, ising.delta]]
        # Solve time evolution:
        output_ising = mesolve(H, psi_0, times, ising.noise_model, [observable])
        plt.plot(times,output_ising.expect[0])

    if model == 'XY':
        XY = pasqalobj.XYPasqalObj(coords, delta, omega, noise_model='dephasing')
        H0, HX, HZ = XY.build_hamiltonian(coords, C3 = 30)
        H = [H0, [HX, XY.omega], [HZ, XY.delta]]
        # Solve time evolution:
        output_XY = mesolve(H, psi_0, times, XY.noise_model, [observable])
        plt.plot(times,output_XY.expect[0])

    # TO DO: Abstract output as `return`
