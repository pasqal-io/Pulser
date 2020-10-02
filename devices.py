from abc import ABC, abstractmethod

from scipy.spatial.distance import pdist
import numpy as np

PLANCK = 6.625 * 10**(-34) # Planck's Constant

class PasqalDevice():
    """
    Abstract class for Pasqal Devices
    """
    def __init__(self):
        pass

    #Physical Constraints
    #def set_couplings(self, atoms)

    def check_array(self,atoms):
        distances = pdist(atoms)
        if len(atoms) > self.max_atoms:
            raise ValueError("Too many atoms")
        if np.max(distances) > self.diameter:
            raise ValueError("Maximal Diameter Error")
        if np.min(distances) < self.min_radius:
            raise ValueError("Minimal Radius Error")

class Device1(PasqalDevice):
    """
    A Pasqal Devices well suited for Ising model
    """
    def __init__(self, atoms):
        super().__init__()
        # Spatial Constraints
        self.max_atoms = 100
        self.diameter = 100
        self.min_radius = 4
        self.check_array(atoms)
        self.atom_array = atoms
        print(f"DEVICE STARTED. ATOM COORDINATES ARE \n {atoms}")


class Device2(PasqalDevice):
    """
    A Pasqal Devices well suited for Ising model
    """
    def __init__(self, atoms):
        # Spatial Constraints
        self.max_atoms = 1000
        self.diameter = 100
        self.min_radius = 4



    #Physical Constraints
    #def set_couplings(self, atoms)
