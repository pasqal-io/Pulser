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
"""Defines the configuration of an array of neutral atoms."""

from __future__ import annotations

from collections.abc import Mapping, Iterable
import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from typing import Any, cast, Optional, Union

import pulser
from pulser.json.utils import obj_to_dict

QubitId = Union[int, str]


class Register:
    """A quantum register containing a set of qubits.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates (in μm) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(self, qubits: Mapping[Any, ArrayLike]):
        """Initializes a custom Register."""
        if not isinstance(qubits, dict):
            raise TypeError(
                "The qubits have to be stored in a dictionary "
                "matching qubit ids to position coordinates."
            )
        if not qubits:
            raise ValueError(
                "Cannot create a Register with an empty qubit " "dictionary."
            )
        self._ids = list(qubits.keys())
        coords = [np.array(v, dtype=float) for v in qubits.values()]
        self._dim = coords[0].size
        if any(c.shape != (self._dim,) for c in coords) or (
            self._dim != 2 and self._dim != 3
        ):
            raise ValueError(
                "All coordinates must be specified as vectors of"
                " size 2 or 3."
            )
        self._coords = coords

    @property
    def qubits(self) -> dict[QubitId, np.ndarray]:
        """Dictionary of the qubit names and their position coordinates."""
        return dict(zip(self._ids, self._coords))

    @classmethod
    def from_coordinates(
        cls,
        coords: np.ndarray,
        center: bool = True,
        prefix: Optional[str] = None,
    ) -> Register:
        """Creates the register from an array of coordinates.

        Args:
            coords (ndarray): The coordinates of each qubit to include in the
                register.

        Keyword args:
            center(defaut=True): Whether or not to center the entire array
                around the origin.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register: A register with qubits placed on the given coordinates.
        """
        if center:
            coords = coords - np.mean(coords, axis=0)  # Centers the array
        if prefix is not None:
            pre = str(prefix)
            qubits = {pre + str(i): pos for i, pos in enumerate(coords)}
        else:
            qubits = dict(cast(Iterable, enumerate(coords)))
        return cls(qubits)

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

        coords = (
            np.array(
                [(x, y) for y in range(rows) for x in range(columns)],
                dtype=float,
            )
            * spacing
        )

        return cls.from_coordinates(coords, center=True, prefix=prefix)

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

        coords = np.array(
            [(x, y) for y in range(rows) for x in range(atoms_per_row)],
            dtype=float,
        )
        coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
        coords[:, 1] *= np.sqrt(3) / 2
        coords *= spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def _hexagon_helper(
        cls,
        layers: int,
        atoms_left: int,
        spacing: float,
        prefix: Optional[str] = None,
    ) -> Register:
        """Helper function for building hexagonal arrays.

        Args:
            layers (int): Number of full layers around a central atom.
            atoms_left (int): Number of atoms on the external layer.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register: A register with qubits placed in a hexagonal layout
                with extra atoms on the outermost layer if needed.
        """
        # y coordinates of the top vertex of a triangle
        crest_y = np.sqrt(3) / 2.0

        # Coordinates of vertices
        start_x = [-1.0, -0.5, 0.5, 1.0, 0.5, -0.5]
        start_y = [0.0, crest_y, crest_y, 0, -crest_y, -crest_y]

        # Steps to place atoms, starting from a vertex
        delta_x = [0.5, 1.0, 0.5, -0.5, -1.0, -0.5]
        delta_y = [crest_y, 0.0, -crest_y, -crest_y, 0.0, crest_y]

        coords = np.array(
            [
                (
                    start_x[side] * layer + atom * delta_x[side],
                    start_y[side] * layer + atom * delta_y[side],
                )
                for layer in range(1, layers + 1)
                for side in range(6)
                for atom in range(1, layer + 1)
            ],
            dtype=float,
        )

        if atoms_left > 0:
            layer = layers + 1
            min_atoms_per_side = atoms_left // 6
            # Extra atoms after balancing all sides
            atoms_left %= 6

            # Order for placing left atoms
            # Top-Left, Top-Right, Bottom (C3 symmetry)...
            # ...Top, Bottom-Right, Bottom-Left (C6 symmetry)
            sides_order = [0, 3, 1, 4, 2, 5]

            coords2 = np.array(
                [
                    (
                        start_x[side] * layer + atom * delta_x[side],
                        start_y[side] * layer + atom * delta_y[side],
                    )
                    for side in range(6)
                    for atom in range(
                        1,
                        min_atoms_per_side + 2
                        if atoms_left > sides_order[side]
                        else min_atoms_per_side + 1,
                    )
                ],
                dtype=float,
            )

            coords = np.concatenate((coords, coords2))

        coords *= spacing
        coords = np.concatenate(([(0.0, 0.0)], coords))

        return cls.from_coordinates(coords, center=False, prefix=prefix)

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

        return cls._hexagon_helper(layers, 0, spacing, prefix)

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

        if n_qubits < 7:
            crest_y = np.sqrt(3) / 2.0
            hex_coords = np.array(
                [
                    (0.0, 0.0),
                    (-0.5, crest_y),
                    (0.5, crest_y),
                    (1.0, 0.0),
                    (0.5, -crest_y),
                    (-0.5, -crest_y),
                ]
            )
            return cls.from_coordinates(
                spacing * hex_coords[:n_qubits], prefix=prefix, center=False
            )

        full_layers = int((-3.0 + np.sqrt(9 + 12 * (n_qubits - 1))) / 6.0)
        atoms_left = n_qubits - 1 - (full_layers ** 2 + full_layers) * 3

        return cls._hexagon_helper(full_layers, atoms_left, spacing, prefix)

    def rotate(self, degrees: float) -> None:
        """Rotates the array around the origin by the given angle.

        Args:
            degrees (float): The angle of rotation in degrees.
        """
        if self._dim != 2:
            raise NotImplementedError("Can only rotate arrays in 2D.")
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

        Note:
            When drawing half the blockade radius, we say there is a blockade
            effect between atoms whenever their respective circles overlap.
            This representation is preferred over drawing the full Rydberg
            radius because it helps in seeing the interactions between atoms.
        """
        # Check dimensions
        if self._dim != 2:
            raise NotImplementedError("Can only draw register layouts in 2D.")

        # Check spacing
        if blockade_radius is not None and blockade_radius <= 0.0:
            raise ValueError(
                "Blockade radius (`blockade_radius` ="
                f" {blockade_radius})"
                " must be greater than 0."
            )

        pos = np.array(self._coords)
        diffs = np.max(pos, axis=0) - np.min(pos, axis=0)
        diffs[diffs < 9] *= 1.5
        diffs[diffs < 9] += 2
        if blockade_radius and draw_half_radius:
            diffs[diffs < blockade_radius] = blockade_radius
        big_side = max(diffs)
        proportions = diffs / big_side
        Ls = proportions * min(
            big_side / 4, 10
        )  # Figsize is, at most, (10,10)

        fig, ax = plt.subplots(figsize=Ls)
        ax.scatter(pos[:, 0], pos[:, 1], s=30, alpha=0.7, c="darkgreen")

        ax.set_xlabel("µm")
        ax.set_ylabel("µm")
        ax.axis("equal")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        if with_labels:
            for q, coords in zip(self._ids, self._coords):
                ax.annotate(q, coords, fontsize=12, ha="left", va="bottom")

        if draw_half_radius:
            if blockade_radius is None:
                raise ValueError("Define 'blockade_radius' to draw.")
            if len(pos) == 1:
                raise NotImplementedError(
                    "Needs more than one atom to draw " "the blockade radius."
                )

            for p in pos:
                circle = plt.Circle(
                    tuple(p), blockade_radius / 2, alpha=0.1, color="darkgreen"
                )
                ax.add_patch(circle)
        if draw_graph and blockade_radius is not None:
            epsilon = 1e-9  # Accounts for rounding errors
            edges = KDTree(pos).query_pairs(blockade_radius * (1 + epsilon))
            lines = pos[(tuple(edges),)]
            lc = mc.LineCollection(lines, linewidths=0.6, colors="grey")
            ax.add_collection(lc)

        else:
            # Only draw central axis lines when not drawing the graph
            ax.axvline(0, c="grey", alpha=0.5, linestyle=":")
            ax.axhline(0, c="grey", alpha=0.5, linestyle=":")

        plt.show()

    def _to_dict(self) -> dict[str, Any]:
        qs = dict(zip(self._ids, map(np.ndarray.tolist, self._coords)))
        return obj_to_dict(self, qs)
