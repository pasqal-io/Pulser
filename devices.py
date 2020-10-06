import scipy
import numpy as np
from abc import ABC, abstractmethod

from . import Raman, Rydberg


class PasqalDevice(ABC):
    """Abstract class for Pasqal Devices.

    Every Pasqal QPU should be defined as a child class of PasqalDevice, thus
    following this template.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates as values (e.g. {'q0':(2, -1, 0), ...}).
    """

    def __init__(self, qubits):
        if not isinstance(qubits, dict):
            raise TypeError("The qubits have to be stored in a dictionary.")

        self.check_array(list(qubits.values()))
        self._qubits = qubits

    @property
    @abstractmethod
    def name(self):
        """The device name."""
        pass

    @property
    @abstractmethod
    def max_dimensionality(self):
        """Whether it works at most with a 2D or 3D array (returns 2 or 3)."""
        pass

    @property
    @abstractmethod
    def max_atom_num(self):
        """Maximum number of atoms that can be simultaneously trapped."""
        pass

    @property
    @abstractmethod
    def max_radial_distance(self):
        """Maximum allowed distance from the center of the array."""
        pass

    @property
    @abstractmethod
    def min_atom_distance(self):
        """Minimal allowed distance of atoms in the trap (in um)."""
        pass

    @property
    @abstractmethod
    def available_channels(self):
        """Channels available on the device."""
        pass

    @property
    def qubits(self):
        """The dictionary of qubit names and their positions."""
        return dict(self._qubits)

    def check_array(self, atoms):
        if len(atoms) > self.max_atom_num:
            raise ValueError("Too many atoms in the array, accepts at most"
                             "{} atoms.".format(self.max_atom_num))
        for pos in atoms:
            if len(pos) != self.max_dimensionality:
                raise ValueError("All qubit positions must be {}D "
                                 "vectors.".format(self.max_dimensionality))

        distances = scipy.spatial.distance.pdist(atoms)
        if np.min(distances) < self.min_atom_distance:
            raise ValueError("The qubit positions don't respect the minimal "
                             "distance between atoms required by the device.")

        if np.max(np.linalg.norm(atoms, axis=1)) > self.max_radial_distance:
            raise ValueError("All qubits must be at most {}um away from the"
                             "center of the array.".format(
                                                    self.max_radial_distance))


class Chadoq2(PasqalDevice):
    """Chadoq2 device specifications."""

    @property
    def name(self):
        """The device name."""
        return "Chadoq2"

    @property
    @abstractmethod
    def max_dimensionality(self):
        """Whether it works at most with a 2D or 3D array (returns 2 or 3)."""
        return 2

    @property
    @abstractmethod
    def max_atom_num(self):
        """Maximum number of atoms that can be simultaneously trapped."""
        return 100

    @property
    @abstractmethod
    def max_radial_distance(self):
        """Maximum allowed distance from the center of the array (in um)."""
        return 50

    @property
    @abstractmethod
    def min_atom_distance(self):
        """Minimal allowed distance of atoms in the trap (in um)."""
        return 4

    @property
    def available_channels(self):
        """Channels available on the device."""
        return {'rydberg_global': Rydberg('global', 50, 1.25),
                'rydberg_local': Rydberg('local', 50, 10),
                'raman_local': Raman('local', 50, 10)}
