import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.distance import pdist

from channels import Raman, Rydberg


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
    def channels(self):
        """Channels available on the device."""
        pass

    @property
    def supported_bases(self):
        """Available electronic transitions for control and measurement."""
        return {ch.basis for ch in self.channels.values()}

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

        if len(atoms) > 1:
            distances = pdist(atoms)  # Pairwise distance between atoms
            if np.min(distances) < self.min_atom_distance:
                raise ValueError("Qubit positions don't respect the minimal "
                                 "distance between atoms for this device.")

        if np.max(np.linalg.norm(atoms, axis=1)) > self.max_radial_distance:
            raise ValueError("All qubits must be at most {}um away from the "
                             "center of the array.".format(
                                                    self.max_radial_distance))

    @classmethod
    def rectangle(cls, rows, columns, spacing=4, prefix=None):
        """Initializes the device with the qubits in a rectangular array.

        Args:
            rows (int): Number of rows.
            columns (int): Number of columns.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in um.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)
        """
        coords = np.array([(x, y) for x in range(columns)
                           for y in range(rows)], dtype=float) * spacing

        return cls(cls._coords_to_qubits(coords, prefix=prefix))

    @classmethod
    def square(cls, side, spacing=4, prefix=None):
        """Initializes the device with the qubits in a square array.

        Args:
            side (int): Side of the square in number of qubits.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in um.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
        """
        return cls.rectangle(side, side, spacing=spacing, prefix=prefix)

    @classmethod
    def triangular_lattice(cls, rows, atoms_per_row, spacing=4, prefix=None):
        """Initializes the device with the qubits in a triangular lattice.

        Initializes the qubits in a triangular lattice pattern, more
        specifically a triangular lattice with horizontal rows, meaning the
        triangles are pointing up and down.

        Args:
            rows (int): Number of rows.
            atoms_per_row (int): Number of atoms per row.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in um.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
        """
        coords = np.array([(x, y) for x in range(atoms_per_row)
                           for y in range(rows)], dtype=float)
        coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
        coords[:, 1] *= np.sqrt(3) / 2
        coords *= spacing

        return cls(cls._coords_to_qubits(coords, prefix=prefix))

    @staticmethod
    def _coords_to_qubits(coords, prefix=None):
        """Centers a coords array and returns the respective qubits dict."""
        coords -= np.mean(coords, axis=0)      # Centers the array
        if prefix is not None:
            pre = str(prefix)
            return {pre+str(i): pos for i, pos in enumerate(coords)}
        else:
            return dict(enumerate(coords))


class Chadoq2(PasqalDevice):
    """Chadoq2 device specifications."""

    @property
    def name(self):
        """The device name."""
        return "Chadoq2"

    @property
    def max_dimensionality(self):
        """Whether it works at most with a 2D or 3D array (returns 2 or 3)."""
        return 2

    @property
    def max_atom_num(self):
        """Maximum number of atoms that can be simultaneously trapped."""
        return 100

    @property
    def max_radial_distance(self):
        """Maximum allowed distance from the center of the array (in um)."""
        return 50

    @property
    def min_atom_distance(self):
        """Minimal allowed distance of atoms in the trap (in um)."""
        return 4

    @property
    def channels(self):
        """Channels available on the device."""
        return {'rydberg_global': Rydberg('global', 50, 2.5),
                'rydberg_local': Rydberg('local', 50, 10, retarget_time=100),
                'rydberg_local2': Rydberg('local', 50, 10, retarget_time=100),
                'raman_local': Raman('local', 50, 10, retarget_time=100)}
