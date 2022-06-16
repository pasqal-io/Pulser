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
from __future__ import annotations

import numpy as np


def square_rect(rows: int, columns: int) -> np.ndarray:
    """A square lattice pattern in a rectangular shape.

    Args:
        rows: Number of rows.
        columns: Number of columns.

    Returns:
        The coordinates of the points in the pattern.
    """
    points = np.mgrid[:columns, :rows].transpose().reshape(-1, 2)
    # Centering
    points = points - np.ceil([columns / 2, rows / 2]) + 1
    return points


def triangular_rect(rows: int, columns: int) -> np.ndarray:
    """A triangular lattice pattern in a rectangular shape.

    Args:
        rows: Number of rows.
        columns: Number of columns.

    Returns:
        The coordinates of the points in the pattern.
    """
    points = square_rect(rows, columns)
    points[:, 0] += 0.5 * np.mod(points[:, 1], 2)
    points[:, 1] *= np.sqrt(3) / 2
    return points


def triangular_hex(n_points: int) -> np.ndarray:
    """A triangular lattice pattern in an hexagonal shape.

    Args:
        n_points: The number of points in the pattern.


    Returns:
        The coordinates of the points in the pattern.
    """
    # y coordinates of the top vertex of a triangle
    crest_y = np.sqrt(3) / 2.0

    if n_points < 7:
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
        return hex_coords[:n_points]

    layers = int((-3.0 + np.sqrt(9 + 12 * (n_points - 1))) / 6.0)
    points_left = n_points - 1 - (layers**2 + layers) * 3

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

    if points_left > 0:
        layer = layers + 1
        min_atoms_per_side = points_left // 6
        # Extra atoms after balancing all sides
        points_left %= 6

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
                    if points_left > sides_order[side]
                    else min_atoms_per_side + 1,
                )
            ],
            dtype=float,
        )

        coords = np.concatenate((coords, coords2))

    coords = np.concatenate((np.zeros((1, 2)), coords))
    return coords
