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
"""Defines the configuration of an array of neutral atoms in 2D."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import pulser
import pulser.register._patterns as patterns
from pulser.register._reg_drawer import RegDrawer
from pulser.register.base_register import BaseRegister, QubitId


class Register(BaseRegister, RegDrawer):
    """A 2D quantum register containing a set of qubits.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates (in μm) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(self, qubits: Mapping[Any, ArrayLike], **kwargs: Any):
        """Initializes a custom Register."""
        super().__init__(qubits, **kwargs)
        if any(c.shape != (self._dim,) for c in self._coords) or (
            self._dim != 2
        ):
            raise ValueError(
                "All coordinates must be specified as vectors of size 2."
            )

    @classmethod
    def square(
        cls, side: int, spacing: float = 4.0, prefix: Optional[str] = None
    ) -> Register:
        """Initializes the register with the qubits in a square array.

        Args:
            side (int): Side of the square in number of qubits.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register: A register with qubits placed in a square array.
        """
        # Check side
        if side < 1:
            raise ValueError(
                f"The number of atoms per side (`side` = {side})"
                " must be greater than or equal to 1."
            )

        return cls.rectangle(side, side, spacing=spacing, prefix=prefix)

    @classmethod
    def rectangle(
        cls,
        rows: int,
        columns: int,
        spacing: float = 4.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Initializes the register with the qubits in a rectangular array.

        Args:
            rows (int): Number of rows.
            columns (int): Number of columns.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)

        Returns:
            Register: A register with qubits placed in a rectangular array.
        """
        # Check rows
        if rows < 1:
            raise ValueError(
                f"The number of rows (`rows` = {rows})"
                " must be greater than or equal to 1."
            )

        # Check columns
        if columns < 1:
            raise ValueError(
                f"The number of columns (`columns` = {columns})"
                " must be greater than or equal to 1."
            )

        # Check spacing
        if spacing <= 0.0:
            raise ValueError(
                f"Spacing between atoms (`spacing` = {spacing})"
                " must be greater than 0."
            )

        coords = patterns.square_rect(rows, columns) * spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def triangular_lattice(
        cls,
        rows: int,
        atoms_per_row: int,
        spacing: float = 4.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Initializes the register with the qubits in a triangular lattice.

        Initializes the qubits in a triangular lattice pattern, more
        specifically a triangular lattice with horizontal rows, meaning the
        triangles are pointing up and down.

        Args:
            rows (int): Number of rows.
            atoms_per_row (int): Number of atoms per row.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register: A register with qubits placed in a triangular lattice.
        """
        # Check rows
        if rows < 1:
            raise ValueError(
                f"The number of rows (`rows` = {rows})"
                " must be greater than or equal to 1."
            )

        # Check atoms per row
        if atoms_per_row < 1:
            raise ValueError(
                "The number of atoms per row"
                f" (`atoms_per_row` = {atoms_per_row})"
                " must be greater than or equal to 1."
            )

        # Check spacing
        if spacing <= 0.0:
            raise ValueError(
                f"Spacing between atoms (`spacing` = {spacing})"
                " must be greater than 0."
            )

        coords = patterns.triangular_rect(rows, atoms_per_row) * spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def hexagon(
        cls, layers: int, spacing: float = 4.0, prefix: Optional[str] = None
    ) -> Register:
        """Initializes the register with the qubits in a hexagonal layout.

        Args:
            layers (int): Number of layers around a central atom.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register: A register with qubits placed in a hexagonal layout.
        """
        # Check layers
        if layers < 1:
            raise ValueError(
                f"The number of layers (`layers` = {layers})"
                " must be greater than or equal to 1."
            )

        # Check spacing
        if spacing <= 0.0:
            raise ValueError(
                f"Spacing between atoms (`spacing` = {spacing})"
                " must be greater than 0."
            )

        n_atoms = 1 + 3 * (layers**2 + layers)
        coords = patterns.triangular_hex(n_atoms) * spacing

        return cls.from_coordinates(coords, center=False, prefix=prefix)

    @classmethod
    def max_connectivity(
        cls,
        n_qubits: int,
        device: pulser.devices._device_datacls.Device,
        spacing: float = None,
        prefix: str = None,
    ) -> Register:
        """Initializes the register with maximum connectivity for a given device.

        In order to maximize connectivity, the basic pattern is the triangle.
        Atoms are first arranged as layers of hexagons around a central atom.
        Extra atoms are placed in such a manner that C3 and C6 rotational
        symmetries are enforced as often as possible.

        Args:
            n_qubits (int): Number of qubits.
            device (Device): The device whose constraints must be obeyed.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
                If omitted, the minimal distance for the device is used.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register: A register with qubits placed for maximum connectivity.
        """
        # Check device
        if not isinstance(device, pulser.devices._device_datacls.Device):
            raise TypeError(
                "'device' must be of type 'Device'. Import a valid"
                " device from 'pulser.devices'."
            )

        # Check number of qubits (1 or above)
        if n_qubits < 1:
            raise ValueError(
                f"The number of qubits (`n_qubits` = {n_qubits})"
                " must be greater than or equal to 1."
            )

        # Check number of qubits (less than the max number of atoms)
        if n_qubits > device.max_atom_num:
            raise ValueError(
                f"The number of qubits (`n_qubits` = {n_qubits})"
                " must be less than or equal to the maximum"
                " number of atoms supported by this device"
                f" ({device.max_atom_num})."
            )

        # Default spacing or check minimal distance
        if spacing is None:
            spacing = device.min_atom_distance
        elif spacing < device.min_atom_distance:
            raise ValueError(
                f"Spacing between atoms (`spacing = `{spacing})"
                " must be greater than or equal to the minimal"
                " distance supported by this device"
                f" ({device.min_atom_distance})."
            )

        coords = patterns.triangular_hex(n_qubits) * spacing

        return cls.from_coordinates(coords, center=False, prefix=prefix)

    def rotate(self, degrees: float) -> None:
        """Rotates the array around the origin by the given angle.

        Args:
            degrees (float): The angle of rotation in degrees.
        """
        if self._layout_info is not None:
            raise TypeError(
                "A register defined from a RegisterLayout cannot be rotated."
            )
        theta = np.deg2rad(degrees)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        self._coords = [rot @ v for v in self._coords]

    def draw(
        self,
        with_labels: bool = True,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        fig_name: str = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the entire register.

        Keyword Args:
            with_labels(bool, default=True): If True, writes the qubit ID's
                next to each qubit.
            blockade_radius(float, default=None): The distance (in μm) between
                atoms below the Rydberg blockade effect occurs.
            draw_half_radius(bool, default=False): Whether or not to draw the
                half the blockade radius surrounding each atoms. If `True`,
                requires `blockade_radius` to be defined.
            draw_graph(bool, default=True): Whether or not to draw the
                interaction between atoms as edges in a graph. Will only draw
                if the `blockade_radius` is defined.
            fig_name(str, default=None): The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig(dict, default={}): Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.

        Note:
            When drawing half the blockade radius, we say there is a blockade
            effect between atoms whenever their respective circles overlap.
            This representation is preferred over drawing the full Rydberg
            radius because it helps in seeing the interactions between atoms.
        """
        super()._draw_checks(
            len(self._ids),
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
        )
        pos = np.array(self._coords)
        fig, ax = self._initialize_fig_axes(
            pos,
            blockade_radius=blockade_radius,
            draw_half_radius=draw_half_radius,
        )
        super()._draw_2D(
            ax,
            pos,
            self._ids,
            with_labels=with_labels,
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
        )
        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def _to_dict(self) -> dict[str, Any]:
        return super()._to_dict()

    def _to_abstract_repr(self) -> dict[str, dict[QubitId, dict[str, float]]]:
        not_str = [id for id in self._ids if not isinstance(id, str)]
        names = [str(id) for id in self._ids]
        if not_str:
            warnings.warn(
                "Register serialization to an abstract representation "
                "irreversibly converts all qubit ID's to strings.",
                stacklevel=7,
            )
            if len(set(names)) < len(names):
                collisions = [id for id in not_str if str(id) in self._ids]
                # TODO: Write serialization error with the name collisions
        return {
            name: {"x": x, "y": y} for name, (x, y) in zip(names, self._coords)
        }
