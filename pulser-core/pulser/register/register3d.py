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
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

import pulser.math as pm
from pulser.json.abstract_repr.deserializer import (
    deserialize_abstract_register,
)
from pulser.json.utils import stringify_qubit_ids
from pulser.register._reg_drawer import RegDrawer
from pulser.register.base_register import BaseRegister, QubitId
from pulser.register.register import Register


class Register3D(BaseRegister, RegDrawer):
    """A 3D quantum register containing a set of qubits.

    Args:
        qubits: Dictionary with the qubit names as keys and their
            position coordinates (in μm) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(
        self,
        qubits: Mapping[Any, ArrayLike | pm.TensorLike],
        **kwargs: Any,
    ):
        """Initializes a custom Register."""
        super().__init__(qubits, **kwargs)
        if (
            any(c.shape != (self.dimensionality,) for c in self._coords_arr)
            or self.dimensionality != 3
        ):
            raise ValueError(
                "All coordinates must be specified as vectors of size 3."
            )

    @classmethod
    def cubic(
        cls,
        side: int,
        spacing: float | pm.TensorLike = 4.0,
        prefix: Optional[str] = None,
    ) -> Register3D:
        """Initializes the register with the qubits in a cubic array.

        Args:
            side: Side of the cube in number of qubits.
            spacing: The distance between neighbouring qubits in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            A 3D register with qubits placed in a cubic array.
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
        spacing: float | pm.TensorLike = 4.0,
        prefix: Optional[str] = None,
    ) -> Register3D:
        """Initializes the register with the qubits in a cuboid array.

        Args:
            rows: Number of rows.
            columns: Number of columns.
            layers: Number of layers.
            spacing: The distance between neighbouring qubits in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)

        Returns:
            A 3D register with qubits placed in a cuboid array.
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
        spacing_ = pm.AbstractArray(spacing)
        if spacing_ <= 0.0:
            raise ValueError(
                f"Spacing between atoms (`spacing` = {spacing})"
                " must be greater than 0."
            )

        coords = (
            pm.AbstractArray(
                [
                    (x, y, z)
                    for z in range(layers)
                    for y in range(rows)
                    for x in range(columns)
                ],
                dtype=float,
            )
            * spacing_
        )

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    def to_2D(self, tol_width: float = 0.0) -> Register:
        """Converts a Register3D into a Register (if possible).

        Args:
            tol_width: The allowed transverse width of the register to be
                projected.

        Returns:
            Returns a 2D register with the coordinates of the atoms
            in a plane, if they are coplanar.

        Raises:
            ValueError: If the atoms are not coplanar.
        """
        coords = self._coords_arr.as_array(detach=True)
        barycenter = coords.sum(axis=0) / coords.shape[0]
        # run SVD
        _, _, vh = np.linalg.svd(coords - barycenter)
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
            coords_2D = pm.vstack(
                [
                    pm.hstack([pm.dot(e_x, r), pm.dot(e_y, r)])
                    for r in self._coords_arr
                ]
            )
            return Register.from_coordinates(coords_2D, labels=self._ids)

    def draw(
        self,
        with_labels: bool = False,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        qubit_colors: Mapping[QubitId, str] = dict(),
        projection: bool = False,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
    ) -> None:
        """Draws the entire register.

        Args:
            with_labels: If True, writes the qubit ID's
                next to each qubit.
            blockade_radius: The distance (in μm) between
                atoms below the Rydberg blockade effect occurs.
            draw_half_radius: Whether or not to draw the
                half the blockade radius surrounding each atoms. If `True`,
                requires `blockade_radius` to be defined.
            draw_graph: Whether or not to draw the
                interaction between atoms as edges in a graph. Will only draw
                if the `blockade_radius` is defined.
            qubit_colors: By default, atoms are drawn with a common default
                color. If this parameter is present, it replaces the colors
                for the specified atoms. Non-specified ones are stilled colored
                with the default value.
            projection: Whether to draw a 2D projection
                instead of a perspective view.
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
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

        pos = self._coords_arr.as_array(detach=True)

        self._draw_3D(
            pos,
            self._ids,
            projection=projection,
            with_labels=with_labels,
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
            qubit_colors=qubit_colors,
        )

        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)
        plt.show()

    def _to_dict(self) -> dict[str, Any]:
        return super()._to_dict()

    def _to_abstract_repr(self) -> list[dict[str, QubitId | float]]:
        names = stringify_qubit_ids(self._ids)
        return [
            {"name": name, "x": x, "y": y, "z": z}
            for name, (x, y, z) in zip(names, self._coords)
        ]

    @staticmethod
    def from_abstract_repr(obj_str: str) -> Register3D:
        """Deserialize a 3D register from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the register encoded
                in the abstract JSON format.
        """
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized register must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )
        return deserialize_abstract_register(obj_str, expected_dim=3)
