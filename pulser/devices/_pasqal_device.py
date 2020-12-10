# Copyright 2020 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.distance import pdist

from pulser.channels import Channel
from pulser.register import Register


@dataclass(frozen=True, repr=False)
class PasqalDevice:
    """Definition of a Pasqal Device.

    Args:
        qubits (dict, Register): A dictionary or a Register class instance with
            all the qubits' names and respective positions in the array.
    """
    name: str
    max_dimensionality: int
    max_atom_num: int
    max_radial_distance: int
    min_atom_distance: int
    channel_names: Tuple[str]
    channel_objs: Tuple[Channel]

    @property
    def channels(self):
        """Available channels on this device."""
        return dict(zip(self.channel_names, self.channel_objs))

    @property
    def supported_bases(self):
        """Available electronic transitions for control and measurement."""
        return {ch.basis for ch in self.channels.values()}

    def specs(self):
        """Prints the device specifications."""
        title = f"{self.name} Specifications"
        lines = [
            "-"*len(title),
            title,
            "-"*len(title),
            "\nRegister requirements:",
            f" - Dimensions: {self.max_dimensionality}D",
            f" - Maximum number of atoms: {self.max_atom_num}",
            f" - Maximum distance from origin: {self.max_radial_distance} um",
            (" - Minimum distance between neighbouring atoms: "
             + f"{self.min_atom_distance} um"),
            "\nChannels:"
            ]
        for name, ch in zip(self.channel_names, self.channel_objs):
            lines.append(f" - '{name}': {ch!r}")

        print("\n".join(lines))

    def __repr__(self):
        return self.name

    def _validate_register(self, register):
        """Checks if 'register' is compatible with this device."""
        if not isinstance(register, Register):
            raise TypeError("register has to be a pulser.Register instance.")

        atoms = list(register.qubits.values())
        if len(atoms) > self.max_atom_num:
            raise ValueError("Too many atoms in the array, the device accepts "
                             "at most {} atoms.".format(self.max_atom_num))
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
            raise ValueError("All qubits must be at most {} um away from the "
                             "center of the array.".format(
                                                    self.max_radial_distance))
