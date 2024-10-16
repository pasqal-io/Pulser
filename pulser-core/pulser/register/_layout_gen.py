# Copyright 2024 Pulser Development Team
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
from scipy.spatial.distance import cdist


def generate_layout_trap_coordinates(
    atom_coords: np.ndarray,
    min_trap_dist: float,
    max_radial_dist: int,
    max_layout_filling: float,
    optimal_layout_filling: float | None = None,
    mesh_resolution: float = 1.0,
    min_traps: int = 1,
    max_traps: int | None = None,
) -> list[np.ndarray]:
    """Generates layout traps for a collection of atomic coordinates.

    Args:
        atom_coords: The coordinates where atoms will be placed.
        device: The Pulser device on which the layout and register
            will be implemented.
        min_trap_dist: The minimum distance between traps.
        max_radial_dist: The maximum distance from the origin.
        max_layout_filling: The maximum ratio of atoms to traps.
        optimal_layout_filling: An optional value for the optimal ratio of
            atoms to traps. If not given, takes max_layout_filling.
        mesh_resolution: The spacing between points in the mesh of candidate
            coordinates, in µm.
        min_traps: The minimum number of traps in the resulting layout.
        max_traps: The maximum number of traps in the resulting layout.
    """
    optimal_layout_filling = optimal_layout_filling or max_layout_filling

    # Generate all coordinates where a trap can be placed
    lx = 2 * max_radial_dist
    side = np.linspace(0, lx, num=int(lx / mesh_resolution)) - max_radial_dist
    x, y = np.meshgrid(side, side)
    in_circle = x**2 + y**2 <= max_radial_dist**2
    coords = np.c_[x[in_circle].ravel(), y[in_circle].ravel()]

    # Get the atoms in the register (the "seeds")
    seeds: list[np.ndarray] = list(atom_coords)
    N_seeds = len(seeds)

    # Record indices and distances between coords and seeds
    c_indx = np.arange(len(coords))
    all_dists = cdist(coords, seeds)

    # Accounts for the case when the needed number is less than min_traps
    min_traps = max(
        np.ceil(N_seeds / max_layout_filling).astype(int), min_traps
    )

    # Use max() in case min_traps is larger than the optimal number
    target_traps = max(
        np.ceil(N_seeds / optimal_layout_filling).astype(int),
        min_traps,
    )
    if max_traps:
        target_traps = min(target_traps, max_traps)

    # This is the region where we can still add traps
    region_left = np.all(all_dists > min_trap_dist, axis=1)
    # The traps start out as being just the seeds
    traps = seeds.copy()
    for i in range(N_seeds, target_traps):
        if not np.any(region_left):
            if i < (min_extra_traps := min_traps - N_seeds):
                raise RuntimeError(
                    f"Failed to find a site for {min_extra_traps - i} traps."
                )
            break
        # Select the point in the valid region that is closest to a seed
        selected = c_indx[region_left][
            np.argmin(np.min(all_dists[region_left][:, :N_seeds], axis=1))
        ]
        # Add the selected point to the traps
        traps.append(coords[selected])
        # Add the distances to the new trap
        all_dists = np.append(all_dists, cdist(coords, [traps[-1]]), axis=1)
        region_left *= all_dists[:, -1] > min_trap_dist
    return traps
