from dataclasses import dataclass, InitVar
from typing import ClassVar, Dict, Union

import numpy as np
from scipy.spatial.distance import pdist

from pulser.channels import Channel, Raman, Rydberg
from pulser.register import Register


@dataclass
class PasqalDevice:
    """Definition of a Pasqal Device.

    Args:
        qubits (dict, Register): A dictionary or a Register class instance with
            all the qubits' names and respective positions in the array.
    """
    name: ClassVar[str]
    max_dimensionality: ClassVar[int]
    max_atom_num: ClassVar[int]
    max_radial_distance: ClassVar[int]
    min_atom_distance: ClassVar[int]
    channels: ClassVar[Dict[str, Channel]]
    qubits: InitVar[Union[dict, Register]]

    def __post_init__(self, qubits):
        if isinstance(qubits, dict):
            register = Register(qubits)
        elif isinstance(qubits, Register):
            register = qubits
        else:
            raise TypeError("The qubits' type must be dict or Register.")

        self._check_array(list(register.qubits.values()))
        self._register = register

    @property
    def supported_bases(self):
        """Available electronic transitions for control and measurement."""
        return {ch.basis for ch in self.channels.values()}

    @property
    def qubits(self):
        """The dictionary of qubit names and their positions."""
        return self._register.qubits

    def _check_array(self, atoms):
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


class Chadoq2(PasqalDevice):
    """Chadoq2 device specifications.

    Args:
        qubits (dict, Register): A dictionary or a Register class instance with
            all the qubits' names and respective positions in the array.
    """

    name = "Chadoq2"
    max_dimensionality = 2
    max_atom_num = 100
    max_radial_distance = 50
    min_atom_distance = 4
    channels = {
        "rydberg_global": Rydberg.Global(50, 2.5),
        "rydberg_local": Rydberg.Local(50, 10, 100),
        "rydberg_local2": Rydberg.Local(50, 10, 100),
        "raman_local": Raman.Local(50, 10, 100),
        }
