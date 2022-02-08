# Copyright 2022 Pulser Development Team
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
"""Defines the configuration of an array of neutral atoms in 3D."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from pulser.register._reg_drawer import RegDrawer
from pulser.register.base_register import BaseRegister
from pulser.register.register import Register


class Register3D(BaseRegister, RegDrawer):
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
            len(self._ids),
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

    def _to_dict(self) -> dict[str, Any]:
        return super()._to_dict()
