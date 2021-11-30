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

from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterable
from collections.abc import Sequence as abcSequence
from typing import Any, cast, Optional, Union, TypeVar, Type
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

import pulser
from pulser.json.utils import obj_to_dict

QubitId = Union[int, str]

T = TypeVar("T", bound="BaseRegister")


class BaseRegister(ABC):
    """The abstract class for a register."""

    @abstractmethod
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
        self._coords = [np.array(v, dtype=float) for v in qubits.values()]
        self._dim = 0

    @property
    def qubits(self) -> dict[QubitId, np.ndarray]:
        """Dictionary of the qubit names and their position coordinates."""
        return dict(zip(self._ids, self._coords))

    @classmethod
    def from_coordinates(
        cls: Type[T],
        coords: np.ndarray,
        center: bool = True,
        prefix: Optional[str] = None,
        labels: Optional[abcSequence[QubitId]] = None,
    ) -> T:
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
            labels (ArrayLike): The list of qubit ids. If defined, each qubit
                id will be set to the corresponding value.

        Returns:
            Register: A register with qubits placed on the given coordinates.
        """
        if center:
            coords = coords - np.mean(coords, axis=0)  # Centers the array
        if prefix is not None:
            pre = str(prefix)
            qubits = {pre + str(i): pos for i, pos in enumerate(coords)}
            if labels is not None:
                raise NotImplementedError(
                    "It is impossible to specify a prefix and "
                    "a set of labels at the same time"
                )

        elif labels is not None:
            if len(coords) != len(labels):
                raise ValueError(
                    f"Label length ({len(labels)}) does not"
                    f"match number of coordinates ({len(coords)})"
                )
            qubits = dict(zip(cast(Iterable, labels), coords))
        else:
            qubits = dict(cast(Iterable, enumerate(coords)))
        return cls(qubits)

    @staticmethod
    def _draw_2D(
        ax: plt.axes._subplots.AxesSubplot,
        pos: np.ndarray,
        ids: list,
        plane: tuple = (0, 1),
        with_labels: bool = True,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        masked_qubits: set[QubitId] = set(),
    ) -> None:
        ix, iy = plane

        ax.scatter(pos[:, ix], pos[:, iy], s=30, alpha=0.7, c="darkgreen")

        # Draw square halo around masked qubits
        if masked_qubits:
            mask_pos = []
            for i, c in zip(ids, pos):
                if i in masked_qubits:
                    mask_pos.append(c)
            mask_arr = np.array(mask_pos)
            ax.scatter(
                mask_arr[:, ix],
                mask_arr[:, iy],
                marker="s",
                s=1200,
                alpha=0.2,
                c="black",
            )

        axes = "xyz"

        ax.set_xlabel(axes[ix] + " (µm)")
        ax.set_ylabel(axes[iy] + " (µm)")
        ax.axis("equal")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        if with_labels:
            # Determine which labels would overlap and merge those
            plot_pos = list(pos[:, (ix, iy)])
            plot_ids: list[Union[list, str]] = [[f"{i}"] for i in ids]
            # Threshold distance between points
            epsilon = 1.0e-2 * np.diff(ax.get_xlim())[0]

            i = 0
            bbs = {}
            while i < len(plot_ids):
                r = plot_pos[i]
                j = i + 1
                overlap = False
                # Put in a list all qubits that overlap at position plot_pos[i]
                while j < len(plot_ids):
                    r2 = plot_pos[j]
                    if np.max(np.abs(r - r2)) < epsilon:
                        plot_ids[i] = plot_ids[i] + plot_ids.pop(j)
                        plot_pos.pop(j)
                        overlap = True
                    else:
                        j += 1
                # Sort qubits in plot_ids[i] according to masked status
                plot_ids[i] = sorted(
                    plot_ids[i],
                    key=lambda s: s in [str(q) for q in masked_qubits],
                )
                # Merge all masked qubits
                has_masked = False
                for j in range(len(plot_ids[i])):
                    if plot_ids[i][j] in [str(q) for q in masked_qubits]:
                        plot_ids[i][j:] = [", ".join(plot_ids[i][j:])]
                        has_masked = True
                        break
                # Add a square bracket that encloses all masked qubits
                if has_masked:
                    plot_ids[i][-1] = "[" + plot_ids[i][-1] + "]"
                # Merge what remains
                plot_ids[i] = ", ".join(plot_ids[i])
                bbs[plot_ids[i]] = overlap
                i += 1

            for q, coords in zip(plot_ids, plot_pos):
                bb = (
                    dict(boxstyle="square", fill=False, ec="gray", ls="--")
                    if bbs[q]
                    else None
                )
                v_al = "center" if bbs[q] else "bottom"
                txt = ax.text(
                    coords[0],
                    coords[1],
                    q,
                    ha="left",
                    va=v_al,
                    wrap=True,
                    bbox=bb,
                )
                txt._get_wrap_line_width = lambda: 50.0

        if draw_half_radius and blockade_radius is not None:
            for p in pos:
                circle = plt.Circle(
                    tuple(p[[ix, iy]]),
                    blockade_radius / 2,
                    alpha=0.1,
                    color="darkgreen",
                )
                ax.add_patch(circle)
                ax.autoscale()
        if draw_graph and blockade_radius is not None:
            epsilon = 1e-9  # Accounts for rounding errors
            edges = KDTree(pos).query_pairs(blockade_radius * (1 + epsilon))
            bonds = pos[(tuple(edges),)]
            if len(bonds) > 0:
                lines = bonds[:, :, (ix, iy)]
            else:
                lines = []
            lc = mc.LineCollection(lines, linewidths=0.6, colors="grey")
            ax.add_collection(lc)

        else:
            # Only draw central axis lines when not drawing the graph
            ax.axvline(0, c="grey", alpha=0.5, linestyle=":")
            ax.axhline(0, c="grey", alpha=0.5, linestyle=":")

    @staticmethod
    def _register_dims(
        pos: np.ndarray,
        blockade_radius: Optional[float] = None,
        draw_half_radius: bool = False,
    ) -> np.ndarray:
        """Returns the dimensions of the register to be drawn."""
        diffs = np.ptp(pos, axis=0)
        diffs[diffs < 9] *= 1.5
        diffs[diffs < 9] += 2
        if blockade_radius and draw_half_radius:
            diffs[diffs < blockade_radius] = blockade_radius

        return np.array(diffs)

    def _draw_checks(
        self,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
    ) -> None:
        """Checks common in all register drawings.

        Keyword Args:
            blockade_radius(float, default=None): The distance (in μm) between
                atoms below the Rydberg blockade effect occurs.
            draw_half_radius(bool, default=False): Whether or not to draw the
                half the blockade radius surrounding each atoms. If `True`,
                requires `blockade_radius` to be defined.
            draw_graph(bool, default=True): Whether or not to draw the
                interaction between atoms as edges in a graph. Will only draw
                if the `blockade_radius` is defined.
        """
        # Check spacing
        if blockade_radius is not None and blockade_radius <= 0.0:
            raise ValueError(
                "Blockade radius (`blockade_radius` ="
                f" {blockade_radius})"
                " must be greater than 0."
            )

        if draw_half_radius:
            if blockade_radius is None:
                raise ValueError("Define 'blockade_radius' to draw.")
            if len(self._ids) == 1:
                raise NotImplementedError(
                    "Needs more than one atom to draw " "the blockade radius."
                )


class Register(BaseRegister):
    """A 2D quantum register containing a set of qubits.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates (in μm) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(self, qubits: Mapping[Any, ArrayLike]):
        """Initializes a custom Register."""
        super().__init__(qubits)
        self._dim = self._coords[0].size
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

        coords = (
            np.array(
                [(x, y) for y in range(rows) for x in range(columns)],
                dtype=float,
            )
            * spacing
        )

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
        theta = np.deg2rad(degrees)
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        self._coords = [rot @ v for v in self._coords]

    def _initialize_fig_axes(
        self,
        pos: np.ndarray,
        blockade_radius: Optional[float] = None,
        draw_half_radius: bool = False,
    ) -> tuple[plt.figure.Figure, plt.axes.Axes]:
        """Creates the Figure and Axes for drawing the register."""
        diffs = super()._register_dims(
            pos,
            blockade_radius=blockade_radius,
            draw_half_radius=draw_half_radius,
        )
        big_side = max(diffs)
        proportions = diffs / big_side
        Ls = proportions * min(
            big_side / 4, 10
        )  # Figsize is, at most, (10,10)
        fig, axes = plt.subplots(figsize=Ls)

        return (fig, axes)

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
        qs = dict(zip(self._ids, map(np.ndarray.tolist, self._coords)))
        return obj_to_dict(self, qs)


class Register3D(BaseRegister):
    """A 3D quantum register containing a set of qubits.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates (in μm) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(self, qubits: Mapping[Any, ArrayLike]):
        """Initializes a custom Register."""
        super().__init__(qubits)
        coords = [np.array(v, dtype=float) for v in qubits.values()]
        self._dim = coords[0].size
        if any(c.shape != (self._dim,) for c in coords) or (self._dim != 3):
            raise ValueError(
                "All coordinates must be specified as vectors of size 3."
            )
        self._coords = coords

    @classmethod
    def cubic(
        cls, side: int, spacing: float = 4.0, prefix: Optional[str] = None
    ) -> Register3D:
        """Initializes the register with the qubits in a cubic array.

        Args:
            side (int): Side of the cube in number of qubits.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            Register3D : A 3D register with qubits placed in a cubic array.
        """
        # Check side
        if side < 1:
            raise ValueError(
                f"The number of atoms per side (`side` = {side})"
                " must be greater than or equal to 1."
            )

        return cls.cuboid(side, side, side, spacing=spacing, prefix=prefix)

    @classmethod
    def cuboid(
        cls,
        rows: int,
        columns: int,
        layers: int,
        spacing: float = 4.0,
        prefix: Optional[str] = None,
    ) -> Register3D:
        """Initializes the register with the qubits in a cuboid array.

        Args:
            rows (int): Number of rows.
            columns (int): Number of columns.
            layers (int): Number of layers.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)

        Returns:
            Register3D : A 3D register with qubits placed in a cuboid array.
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

        coords = (
            np.array(
                [
                    (x, y, z)
                    for z in range(layers)
                    for y in range(rows)
                    for x in range(columns)
                ],
                dtype=float,
            )
            * spacing
        )

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    def to_2D(self, tol_width: float = 0.0) -> Register:
        """Converts a Register3D into a Register (if possible).

        Args:
            tol_width (float): The allowed transverse width of
            the register to be projected.

        Returns:
            Register: Returns a 2D register with the coordinates of the atoms
                in a plane, if they are coplanar.

        Raises:
            ValueError: If the atoms are not coplanar.
        """
        coords = np.array(self._coords)

        barycenter = coords.sum(axis=0) / coords.shape[0]
        # run SVD
        u, s, vh = np.linalg.svd(coords - barycenter)
        e_z = vh[2, :]
        perp_extent = [e_z.dot(r) for r in coords]
        width = np.ptp(perp_extent)
        # A set of vector is coplanar if one of the Singular values is 0
        if width > tol_width:
            raise ValueError(
                f"Atoms are not coplanar (`width` = {width:#.2f} µm)"
            )
        else:
            e_x = vh[0, :]
            e_y = vh[1, :]
            coords_2D = np.array(
                [np.array([e_x.dot(r), e_y.dot(r)]) for r in coords]
            )
            return Register.from_coordinates(coords_2D, labels=self._ids)

    def _initialize_fig_axes_projection(
        self,
        pos: np.ndarray,
        blockade_radius: Optional[float] = None,
        draw_half_radius: bool = False,
    ) -> tuple[plt.figure.Figure, plt.axes.Axes]:
        """Creates the Figure and Axes for drawing the register projections."""
        diffs = super()._register_dims(
            pos,
            blockade_radius=blockade_radius,
            draw_half_radius=draw_half_radius,
        )

        proportions = []
        for (ix, iy) in combinations(np.arange(3), 2):
            big_side = max(diffs[[ix, iy]])
            Ls = diffs[[ix, iy]] / big_side
            Ls *= max(
                min(big_side / 4, 10), 4
            )  # Figsize is, at most, (10,10), and, at least (4,*) or (*,4)
            proportions.append(Ls)

        fig_height = np.max([Ls[1] for Ls in proportions])

        max_width = 0
        for i, (width, height) in enumerate(proportions):
            proportions[i] = (width * fig_height / height, fig_height)
            max_width = max(max_width, proportions[i][0])
        widths = [max(Ls[0], max_width / 5) for Ls in proportions]
        fig_width = min(np.sum(widths), fig_height * 4)

        rescaling = 20 / max(max(fig_width, fig_height), 20)
        figsize = (rescaling * fig_width, rescaling * fig_height)

        fig, axes = plt.subplots(
            ncols=3,
            figsize=figsize,
            gridspec_kw=dict(width_ratios=widths),
        )

        return (fig, axes)

    def draw(
        self,
        with_labels: bool = False,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        projection: bool = False,
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
            projection(bool, default=False): Whether to draw a 2D projection
                instead of a perspective view.
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
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
        )

        pos = np.array(self._coords)

        if draw_graph and blockade_radius is not None:
            epsilon = 1e-9  # Accounts for rounding errors
            edges = KDTree(pos).query_pairs(blockade_radius * (1 + epsilon))

        if projection:
            labels = "xyz"
            fig, axes = self._initialize_fig_axes_projection(
                pos,
                blockade_radius=blockade_radius,
                draw_half_radius=draw_half_radius,
            )
            fig.tight_layout(w_pad=6.5)

            for ax, (ix, iy) in zip(axes, combinations(np.arange(3), 2)):
                super()._draw_2D(
                    ax,
                    pos,
                    self._ids,
                    plane=(
                        ix,
                        iy,
                    ),
                    with_labels=with_labels,
                    blockade_radius=blockade_radius,
                    draw_graph=draw_graph,
                    draw_half_radius=draw_half_radius,
                )
                ax.set_title(
                    "Projection onto\n the "
                    + labels[ix]
                    + labels[iy]
                    + "-plane"
                )

        else:
            fig = plt.figure(figsize=2 * plt.figaspect(0.5))

            if draw_graph and blockade_radius is not None:
                bonds = {}
                for i, j in edges:
                    xi, yi, zi = pos[i]
                    xj, yj, zj = pos[j]
                    bonds[(i, j)] = [[xi, xj], [yi, yj], [zi, zj]]

            for i in range(1, 3):
                ax = fig.add_subplot(
                    1, 2, i, projection="3d", azim=-60 * (-1) ** i, elev=15
                )

                ax.scatter(
                    pos[:, 0],
                    pos[:, 1],
                    pos[:, 2],
                    s=30,
                    alpha=0.7,
                    c="darkgreen",
                )

                if with_labels:
                    for q, coords in zip(self._ids, self._coords):
                        ax.text(
                            coords[0],
                            coords[1],
                            coords[2],
                            q,
                            fontsize=12,
                            ha="left",
                            va="bottom",
                        )

                if draw_half_radius and blockade_radius is not None:
                    mesh_num = 20 if len(self._ids) > 10 else 40
                    for r in pos:
                        x0, y0, z0 = r
                        radius = blockade_radius / 2

                        # Strange behavior pf mypy using "imaginary slice step"
                        # u, v = np.pi * np.mgrid[0:2:50j, 0:1:50j]

                        v, u = np.meshgrid(
                            np.arccos(np.linspace(-1, 1, num=mesh_num)),
                            np.linspace(0, 2 * np.pi, num=mesh_num),
                        )
                        x = radius * np.cos(u) * np.sin(v) + x0
                        y = radius * np.sin(u) * np.sin(v) + y0
                        z = radius * np.cos(v) + z0
                        # alpha controls opacity
                        ax.plot_surface(x, y, z, color="darkgreen", alpha=0.1)

                if draw_graph and blockade_radius is not None:
                    for x, y, z in bonds.values():
                        ax.plot(x, y, z, linewidth=1.5, color="grey")

                ax.set_xlabel("x (µm)")
                ax.set_ylabel("y (µm)")
                ax.set_zlabel("z (µm)")

        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()
