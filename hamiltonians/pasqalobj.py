# Playing with the Qobj class from qutip

import numpy as np
from scipy import *
from qutip import *
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Constants
MAXCOORD = 100 # finite size of space
HBAR = 1

class PasqalObj:
    """
    Defines a Hamiltonian compatible with Pasqal processors
    """

    def __init__(self, coords, delta, omega, noise_model='None'):
        self.coords = coords
        N = len(self.coords) # Size of the system

        #if len(self.coords) > 100:
        #    raise ValueError("Number of atoms exceeds Pasqal processor capacity")

        distances = cdist(coords, coords)
        self.rmin = np.min(distances[distances>0])
        self.rmax = np.max(distances)

        print(f"Minimal distance = {self.rmin}")
        print(f"Maximal distance = {self.rmax}")

        if self.rmin < 4:
            raise ValueError("Minimal separation between atoms breached.")
        if self.rmax > 100:
            raise ValueError("Maximal separation between atoms breached.")

        # assign time-dependent functions
        times = np.linspace(0,10**8,10**3)
        self.delta = delta
        if np.max(np.abs(self.delta(times))) > 15:
            raise ValueError("Amplitude of coefficient delta breaches maximum allowed.")

        self.omega = omega
        if np.max(np.abs(self.omega(times))) > 15:
            raise ValueError("Amplitude of coefficient omega breaches maximum allowed.")

        if noise_model == 'dephasing':
            rate = 1/25 # In 1/µs, for Omega ~ 4π Mhz
            P0 = tensor ( [Qobj([[1,0],[0, np.sqrt(1-rate)]]) for _ in range(N)] )
            P1 = tensor ( [Qobj([[0,np.sqrt(rate)],[0, 0]]) for _ in range(N)] )
            self.noise_model = [P0, P1]

        if noise_model == 'relaxation':
            rate = 1/25 # In 1/µs, for Omega ~ 4π Mhz
            P0 = tensor( [qeye(2) * np.sqrt(1-rate) for _ in range(N)] )
            P1 = tensor( [sigmaz() * np.sqrt(rate) for _ in range(N)] )
            self.noise_model = [P0,P1]

        if noise_model == 'None':
            self.noise_model = []



    def rad(self, i, j, coords):
        """Returns the distance between the ith and jth atoms

        Args:
        coords: coordinates of atoms
        """
        return np.sqrt( (coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2 )


class IsingPasqalObj(PasqalObj):
    """Define an Ising hamiltonian compatible with Pasqal processors
    """

    def __init__(self, coords, delta, omega, noise_model='None'):
        PasqalObj.__init__(self, coords, delta, omega, noise_model)


    def build_hamiltonian(self, coords, C6):
        """Constructs the Ising hamiltonian

        args:

        N : number of particles
        coords: coordinates of particles
        C6: coefficient for the Van der Waals interaction
        """
        N = len(coords)

        si = qeye(2)
        sx = sigmax()
        sz = sigmaz()
        nn = 0.5 * (si + sz)

        sx_list = []
        sz_list = []
        nn_list = []

        for j in range(N): # Note these are different for the XY
            op_list = [si for _ in range(N)]

            op_list[j] = sx
            sx_list.append(tensor(op_list))

            op_list[j] = sz
            sz_list.append(tensor(op_list))

            op_list[j] = nn
            nn_list.append(tensor(op_list))


        # construct the hamiltonian
        H0 = 0   # Time Independent part
        HX = 0   # Time dependent part
        HZ = 0   # Time dependent part

        # x,z interactions
        for j in range(N):
            HX += 0.5 * HBAR * sx_list[j]
            HZ += - HBAR * nn_list[j]

        # Ising interaction terms
        for i in range(N):
            for j in range(N):
                if i != j:
                    H0 += C6 * nn_list[i] * nn_list[j] / self.rad(i,j,coords)**6

        return H0, HX, HZ


class XYPasqalObj(PasqalObj):
    """Define an XY hamiltonian compatible with Pasqal processors
    """

    def __init__(self, coords, delta, omega, noise_model='None'):
        PasqalObj.__init__(self, coords, delta, omega, noise_model)


    def build_hamiltonian(self, coords, C3):
        """Constructs the XY hamiltonian

        args:

        N : number of particles
        coords: coordinates of particles
        C3: coefficient for the Dipole-Dipole interaction
        """
        N = len(coords)

        si = qeye(2)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()

        sx_list = []
        sy_list = []
        sz_list = []

        for j in range(N):
            op_list = [si for _ in range(N)]

            op_list[j] = sx
            sx_list.append(tensor(op_list))

            op_list[j] = sy
            sy_list.append(tensor(op_list))

            op_list[j] = sz
            sz_list.append(tensor(op_list))


        # construct the hamiltonian
        H0 = 0   # Time Independent part
        HX = 0   # Time dependent part
        HZ = 0   # Time dependent part

        # x,z interactions
        for j in range(N):
            HX += 0.5 * HBAR  * sx_list[j]
            HZ += - 0.5 * HBAR  * sz_list[j]

        # XY interaction terms
        for i in range(N):
            for j in range(N):
                if i != j:
                    H0 += 2 * C3 * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j])  / self.rad(i,j,coords)**3

        return H0, HX, HZ
