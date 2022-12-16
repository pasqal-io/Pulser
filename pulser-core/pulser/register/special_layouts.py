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
"""Special register layouts defined for convenience."""

from __future__ import annotations

from typing import Any, cast

import pulser.register._patterns as patterns
from pulser.json.utils import obj_to_dict
from pulser.register import Register
from pulser.register.register_layout import RegisterLayout


class SquareLatticeLayout(RegisterLayout):
    """A RegisterLayout with a square lattice pattern in a rectangular shape.

    Args:
        rows: The number of rows of traps.
        columns: The number of columns of traps.
        spacing: The distance between neighbouring traps (in µm).
    """

    def __init__(self, rows: int, columns: int, spacing: int):
        """Initializes a SquareLatticeLayout."""
        self._rows = int(rows)
        self._columns = int(columns)
        self._spacing = int(spacing)
        slug = (
            f"SquareLatticeLayout({self._rows}x{self._columns}, "
            f"{self._spacing}µm)"
        )
        super().__init__(
            patterns.square_rect(self._rows, self._columns) * self._spacing,
            slug=slug,
        )

    def square_register(self, side: int, prefix: str = "q") -> Register:
        """Defines a register with a square shape.

        Args:
            side: The length of the square's side, in number of atoms.
            prefix: The prefix for the qubit ids. Each qubit ID starts
                with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            The register instance created from this layout.
        """
        return self.rectangular_register(side, side, prefix=prefix)

    def rectangular_register(
        self,
        rows: int,
        columns: int,
        prefix: str = "q",
    ) -> Register:
        """Defines a register with a rectangular shape.

        Args:
            rows: The number of rows in the register.
            columns: The number of columns in the register.
            prefix: The prefix for the qubit ids. Each qubit ID starts
                with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            The register instance created from this layout.
        """
        if rows > self._rows or columns > self._columns:
            raise ValueError(
                f"A '{rows}x{columns}' array doesn't fit a "
                f"{self._rows}x{self._columns} SquareLatticeLayout."
            )
        points = patterns.square_rect(rows, columns) * self._spacing
        trap_ids = self.get_traps_from_coordinates(*points)
        qubit_ids = [f"{prefix}{i}" for i in range(len(trap_ids))]
        return cast(
            Register, self.define_register(*trap_ids, qubit_ids=qubit_ids)
        )

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self._rows, self._columns, self._spacing)


class TriangularLatticeLayout(RegisterLayout):
    """A RegisterLayout with a triangular lattice pattern in a hexagonal shape.

    Args:
        n_traps: The number of traps in the layout.
        spacing: The distance between neighbouring traps (in µm).
    """

    def __init__(self, n_traps: int, spacing: int):
        """Initializes a TriangularLatticeLayout."""
        self._spacing = int(spacing)
        slug = f"TriangularLatticeLayout({int(n_traps)}, {self._spacing}µm)"
        super().__init__(
            patterns.triangular_hex(int(n_traps)) * self._spacing, slug=slug
        )

    def hexagonal_register(self, n_atoms: int, prefix: str = "q") -> Register:
        """Defines a register with an hexagonal shape.

        Args:
            n_atoms: The number of atoms in the register.
            prefix: The prefix for the qubit ids. Each qubit ID starts
                with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            The register instance created from this layout.
        """
        if n_atoms > self.number_of_traps:
            raise ValueError(
                f"The desired register has more atoms ({n_atoms}) than there"
                " are traps in this TriangularLatticeLayout"
                f" ({self.number_of_traps})."
            )
        points = patterns.triangular_hex(n_atoms) * self._spacing
        trap_ids = self.get_traps_from_coordinates(*points)
        qubit_ids = [f"{prefix}{i}" for i in range(len(trap_ids))]
        return cast(
            Register, self.define_register(*trap_ids, qubit_ids=qubit_ids)
        )

    def rectangular_register(
        self, rows: int, atoms_per_row: int, prefix: str = "q"
    ) -> Register:
        """Defines a register with a rectangular shape.

        Args:
            rows: The number of rows in the register.
            atoms_per_row: The number of atoms in each row.
            prefix: The prefix for the qubit ids. Each qubit ID starts
                with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).

        Returns:
            The register instance created from this layout.
        """
        if rows * atoms_per_row > self.number_of_traps:
            raise ValueError(
                f"A '{rows}x{atoms_per_row}' rectangular subset of a "
                "triangular lattice has more atoms than there are traps in "
                f"this TriangularLatticeLayout ({self.number_of_traps})."
            )
        points = patterns.triangular_rect(rows, atoms_per_row) * self._spacing
        trap_ids = self.get_traps_from_coordinates(*points)
        qubit_ids = [f"{prefix}{i}" for i in range(len(trap_ids))]
        return cast(
            Register, self.define_register(*trap_ids, qubit_ids=qubit_ids)
        )

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(self, self.number_of_traps, self._spacing)
