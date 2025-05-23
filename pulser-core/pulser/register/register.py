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

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

import pulser
import pulser.math as pm
import pulser.register._patterns as patterns
from pulser.json.abstract_repr.deserializer import (
    deserialize_abstract_register,
)
from pulser.json.utils import stringify_qubit_ids
from pulser.register._layout_gen import generate_trap_coordinates
from pulser.register._reg_drawer import RegDrawer
from pulser.register.base_register import BaseRegister, QubitId

if TYPE_CHECKING:
    from pulser.devices import Device
    from pulser.devices._device_datacls import BaseDevice


class Register(BaseRegister, RegDrawer):
    """A 2D quantum register containing a set of qubits.

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
            or self.dimensionality != 2
        ):
            raise ValueError(
                "All coordinates must be specified as vectors of size 2."
            )

    @classmethod
    def square(
        cls,
        side: int,
        spacing: float | pm.TensorLike = 4.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Creates the register with the qubits in a square array.

        Args:
            side: Side of the square in number of qubits.
            spacing: The distance between neighbouring qubits in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            A register with qubits placed in a square array.
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
        spacing: float | pm.TensorLike = 4.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Creates a rectangular array of qubits on a square lattice.

        Args:
            rows: Number of rows.
            columns: Number of columns.
            spacing: The distance between neighbouring qubits in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)

        Returns:
            A register with qubits placed in a rectangular array.
        """
        return cls.rectangular_lattice(rows, columns, spacing, spacing, prefix)

    @classmethod
    def rectangular_lattice(
        cls,
        rows: int,
        columns: int,
        row_spacing: float | pm.TensorLike = 4.0,
        col_spacing: float | pm.TensorLike = 2.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Creates a rectangular array of qubits on a rectangular lattice.

        Args:
            rows: Number of rows.
            columns: Number of columns.
            row_spacing: The distance between rows in μm.
            col_spacing: The distance between columns in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)

        Returns:
            Register with qubits placed in a rectangular array on a
            rectangular lattice.
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

        row_spacing_ = pm.AbstractArray(row_spacing)
        col_spacing_ = pm.AbstractArray(col_spacing)

        # Check spacing
        if row_spacing_ <= 0.0 or col_spacing_ <= 0.0:
            raise ValueError("Spacing between atoms must be greater than 0.")

        coords = pm.AbstractArray(patterns.square_rect(rows, columns))
        coords[:, 0] = coords[:, 0] * col_spacing_
        coords[:, 1] = coords[:, 1] * row_spacing_

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def triangular_lattice(
        cls,
        rows: int,
        atoms_per_row: int,
        spacing: float | pm.TensorLike = 4.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Creates the register with the qubits in a triangular lattice.

        Initializes the qubits in a triangular lattice pattern, more
        specifically a triangular lattice with horizontal rows, meaning the
        triangles are pointing up and down.

        Args:
            rows: Number of rows.
            atoms_per_row: Number of atoms per row.
            spacing: The distance between neighbouring qubits in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            A register with qubits placed in a triangular lattice.
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

        spacing_ = pm.AbstractArray(spacing)
        # Check spacing
        if spacing_ <= 0.0:
            raise ValueError(
                f"Spacing between atoms (`spacing` = {spacing})"
                " must be greater than 0."
            )

        coords = (
            pm.AbstractArray(patterns.triangular_rect(rows, atoms_per_row))
            * spacing_
        )
        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def hexagon(
        cls,
        layers: int,
        spacing: float | pm.TensorLike = 4.0,
        prefix: Optional[str] = None,
    ) -> Register:
        """Creates the register with the qubits in a hexagonal layout.

        Args:
            layers: Number of layers around a central atom.
            spacing: The distance between neighbouring qubits in μm.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            A register with qubits placed in a hexagonal layout.
        """
        # Check layers
        if layers < 1:
            raise ValueError(
                f"The number of layers (`layers` = {layers})"
                " must be greater than or equal to 1."
            )

        spacing_ = pm.AbstractArray(spacing)
        # Check spacing
        if spacing_ <= 0.0:
            raise ValueError(
                f"Spacing between atoms (`spacing` = {spacing})"
                " must be greater than 0."
            )

        n_atoms = 1 + 3 * (layers**2 + layers)
        coords = pm.AbstractArray(patterns.triangular_hex(n_atoms)) * spacing_

        return cls.from_coordinates(coords, center=False, prefix=prefix)

    @classmethod
    def max_connectivity(
        cls,
        n_qubits: int,
        device: BaseDevice,
        spacing: float | pm.TensorLike | None = None,
        prefix: str | None = None,
    ) -> Register:
        """Initializes the register with maximum connectivity for a device.

        In order to maximize connectivity, the basic pattern is the triangle.
        Atoms are first arranged as layers of hexagons around a central atom.
        Extra atoms are placed in such a manner that C3 and C6 rotational
        symmetries are enforced as often as possible.

        Args:
            n_qubits: Number of qubits.
            device: The device whose constraints must be obeyed.
            spacing: The distance between neighbouring qubits in μm.
                If omitted, the minimal distance for the device is used.
            prefix: The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            A register with qubits placed for maximum connectivity.
        """
        # Check device
        if not isinstance(device, pulser.devices._device_datacls.BaseDevice):
            raise TypeError("'device' must be of type 'BaseDevice'.")

        # Check number of qubits (1 or above)
        if n_qubits < 1:
            raise ValueError(
                f"The number of qubits (`n_qubits` = {n_qubits})"
                " must be greater than or equal to 1."
            )

        # Check number of qubits (less than the max number of atoms)
        if device.max_atom_num is not None and n_qubits > device.max_atom_num:
            raise ValueError(
                f"The number of qubits (`n_qubits` = {n_qubits})"
                " must be less than or equal to the maximum"
                " number of atoms supported by this device"
                f" ({device.max_atom_num})."
            )

        # Default spacing or check minimal distance
        if spacing is None:
            spacing_ = pm.AbstractArray(device.min_atom_distance)
        elif (
            spacing_ := pm.AbstractArray(spacing)
        ) < device.min_atom_distance:
            raise ValueError(
                f"Spacing between atoms (`spacing = `{spacing})"
                " must be greater than or equal to the minimal"
                " distance supported by this device"
                f" ({device.min_atom_distance})."
            )
        if spacing_ <= 0.0:
            # spacing is None or 0.0, device.min_atom_distance is 0.0
            raise NotImplementedError(
                "Maximum connectivity layouts are not well defined for a "
                "device with 'min_atom_distance=0.0'."
            )

        coords = pm.AbstractArray(patterns.triangular_hex(n_qubits)) * spacing_

        return cls.from_coordinates(coords, center=False, prefix=prefix)

    def with_automatic_layout(
        self,
        device: Device,
        layout_slug: str | None = None,
    ) -> Register:
        """Replicates the register with an automatically generated layout.

        The generated `RegisterLayout` can be accessed via `Register.layout`.

        Args:
            device: The device constraints for the layout generation.
            layout_slug: An optional slug for the generated layout.

        Raises:
            RuntimeError: If the automatic layout generation fails to meet
                the device constraints.
            NotImplementedError: When the register has differentiable
                coordinates (ie torch Tensors with requires_grad=True).

        Returns:
            Register: A new register instance with identical qubit IDs and
            coordinates but also the newly generated RegisterLayout.
        """
        if not isinstance(device, pulser.devices.Device):
            raise TypeError(
                f"'device' must be of type Device, not {type(device)}."
            )
        if self._coords_arr.requires_grad:
            raise NotImplementedError(
                "'Register.with_automatic_layout()' does not support "
                "registers with differentiable coordinates."
            )

        max_traps = device.max_layout_traps
        if device.min_layout_filling > 0.0:
            # This is akin to imposing a max number of traps for a given
            # minimum filling and number of qubits
            max_allowed_traps = int(
                len(self.qubit_ids) / device.min_layout_filling
            )
            if max_allowed_traps > device.min_layout_traps:
                # We only enforce min_layout_filling if the maximum number
                # of traps it allows is greater than the minimum number of
                # traps required
                max_traps = min(
                    max_traps
                    or max_allowed_traps,  # In case max_traps is None
                    max_allowed_traps,
                )

        trap_coords = generate_trap_coordinates(
            self.sorted_coords,
            min_trap_dist=device.min_atom_distance,
            max_radial_dist=device.max_radial_distance,
            max_layout_filling=device.max_layout_filling,
            optimal_layout_filling=device.optimal_layout_filling,
            min_traps=device.min_layout_traps,
            max_traps=max_traps,
        )
        layout = pulser.register.RegisterLayout(trap_coords, slug=layout_slug)
        trap_ids = layout.get_traps_from_coordinates(
            *self._coords_arr.as_array()
        )
        return cast(
            Register,
            layout.define_register(*trap_ids, qubit_ids=self.qubit_ids),
        )

    def rotated(self, degrees: float) -> Register:
        """Makes a new rotated register.

        All coordinates are rotated counter-clockwise around the origin.

        Args:
            degrees: The angle of rotation in degrees.

        Returns:
            Register: A new register rotated around the origin by the given
            angle.
        """
        theta = np.deg2rad(degrees)
        rot = pm.vstack(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        if self.layout is not None:
            warnings.warn(
                "The rotated register won't have an associated "
                "'RegisterLayout'.",
                stacklevel=2,
            )

        return Register(
            dict(zip(self.qubit_ids, [rot @ v for v in self._coords_arr]))
        )

    def draw(
        self,
        with_labels: bool = True,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        qubit_colors: Mapping[QubitId, str] = dict(),
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
        custom_ax: Optional[Axes] = None,
        show: bool = True,
        draw_empty_sites: bool = False,
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
            fig_name: The name on which to save the figure.
                If None the figure will not be saved.
            kwargs_savefig: Keywords arguments for
                ``matplotlib.pyplot.savefig``. Not applicable if `fig_name`
                is ``None``.
            custom_ax: If present, instead of creating its own Axes object,
                the function will use the provided one. Warning: if fig_name
                is set, it may save content beyond what is drawn in this
                function.
            show: Whether or not to call `plt.show()` before returning. When
                combining this plot with other ones in a single figure, one may
                need to set this flag to False.
            draw_empty_sites: If True, also draws the sites of the associated
                layout that do not contain an atom.

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

        if draw_empty_sites:
            if self.layout is None:
                raise ValueError(
                    "The register must have an associated RegisterLayout "
                    "to draw the empty sites."
                )
            layout = self.layout
            layout_ids = list(layout.traps_dict.keys())
            filled_traps_ids = layout.get_traps_from_coordinates(
                *tuple(self.qubits.values())
            )
            empty_traps_ids = [
                trap_id
                for trap_id in layout_ids
                if trap_id not in filled_traps_ids
            ]
            empty_traps_reg = self.layout.define_register(
                *empty_traps_ids,
                qubit_ids=[str(trap_id) for trap_id in empty_traps_ids],
            )

        pos = self._coords_arr.as_array(detach=True)
        if custom_ax is None:
            custom_ax = cast(
                plt.Axes,
                self._initialize_fig_axes(
                    layout.sorted_coords if draw_empty_sites else pos,
                    blockade_radius=blockade_radius,
                    draw_half_radius=draw_half_radius,
                )[1],
            )

        draw_kwargs = dict(
            ax=custom_ax,
            blockade_radius=blockade_radius,
            draw_graph=draw_graph,
            draw_half_radius=draw_half_radius,
        )

        if draw_empty_sites:
            super()._draw_2D(
                ids=empty_traps_reg.qubit_ids,
                pos=empty_traps_reg._coords_arr.as_array(detach=True),
                with_labels=False,
                label_name="empty",
                are_traps=True,
                **draw_kwargs,  # type: ignore
            )

        super()._draw_2D(
            ids=self._ids,
            pos=pos,
            qubit_colors=qubit_colors,
            with_labels=with_labels,
            **draw_kwargs,  # type: ignore
        )

        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)

        if show:
            plt.show()

    def _to_dict(self) -> dict[str, Any]:
        return super()._to_dict()

    def _to_abstract_repr(self) -> list[dict[str, Union[QubitId, float]]]:
        names = stringify_qubit_ids(self._ids)
        return [
            {"name": name, "x": x, "y": y}
            for name, (x, y) in zip(names, self._coords_arr.tolist())
        ]

    @staticmethod
    def from_abstract_repr(obj_str: str) -> Register:
        """Deserialize a register from an abstract JSON object.

        Args:
            obj_str (str): the JSON string representing the register encoded
                in the abstract JSON format.
        """
        if not isinstance(obj_str, str):
            raise TypeError(
                "The serialized register must be given as a string. "
                f"Instead, got object of type {type(obj_str)}."
            )
        return deserialize_abstract_register(obj_str, expected_dim=2)
