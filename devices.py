from scipy.spatial.distance import pdist
import numpy as np

PLANCK = 6.625 * 10**(-34) # Planck's Constant

class PasqalDevice():
    """
    Abstract class for Pasqal Devices
    """
    #Physical Constraints
    #def set_couplings(self, atoms)

    def check_array(self,atoms):
        distances = pdist(atoms)
        # check if it can do Ising
        if np.max(distances) > self.diameter:
            raise ValueError("Maximal Diameter Error")

class Device1(PasqalDevice):
    """
    A Pasqal Devices well suited for Ising model
    """
    def __init__(self, atom_array):
        # Spatial Constraints
        self.max_atoms = 100
        self.diameter = 100
        self.min_radius = 4
        self.check_array(atom_array)
        self.atom_array = atom_array


class Device2(PasqalDevice):
    """
    A Pasqal Devices well suited for Ising model
    """
    def __init__(self, atom_array):
        # Spatial Constraints
        self.max_atoms = 1000
        self.diameter = 100
        self.min_radius = 4



    #Physical Constraints
    #def set_couplings(self, atoms)
