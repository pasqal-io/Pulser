# Copyright 2023 Pulser Development Team
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
"""Defines weight maps on top of traps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from pulser.register._reg_drawer import RegDrawer
from pulser.register.traps import COORD_PRECISION

if TYPE_CHECKING:
    from pulser.register.base_register import QubitId


@dataclass
class WeightMap(RegDrawer):
    """Defines a generic map of weights on traps.

    The sum of the provided weights must be equal to 1.

    Args:
        trap_coordinates: An array containing the coordinates of the traps.
        weights: A list weights to associate to the traps.
    """

    trap_coordinates: ArrayLike
    weights: list[float]

    def __post_init__(self) -> None:
        if len(cast(list, self.trap_coordinates)) != len(self.weights):
            raise ValueError("Number of traps and weights don't match.")
        if not np.all(np.array(self.weights) >= 0):
            raise ValueError("All weights must be non-negative.")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-16):
            raise ValueError("The sum of the weights should be 1.")

    def get_qubit_weight_map(
        self, qubits: Mapping[QubitId, np.ndarray]
    ) -> dict[QubitId, float]:
        """Creates a map between qubit IDs and the weight on their sites."""
        qubit_weight_map = {}
        coords_arr = np.array(self.trap_coordinates)
        weights_arr = np.array(self.weights)
        for qid, pos in qubits.items():
            dists = np.round(
                np.linalg.norm(coords_arr - np.array(pos), axis=1),
                decimals=COORD_PRECISION,
            )
            matches = np.argwhere(dists == 0.0)
            qubit_weight_map[qid] = float(np.sum(weights_arr[matches]))
        return qubit_weight_map

    def draw(
        self,
        with_labels: bool = True,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
        custom_ax: Optional[Axes] = None,
        show: bool = True,
    ) -> None:
        """Draws the detuning map.

        Args:
            with_labels: If True, writes the qubit ID's
                next to each qubit.
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
        """
        pos = np.array(self.trap_coordinates)
        if custom_ax is None:
            _, custom_ax = self._initialize_fig_axes(pos)

        super()._draw_2D(
            custom_ax,
            pos,
            [i for i, _ in enumerate(cast(list, self.trap_coordinates))],
            with_labels=with_labels,
            dmm_qubits=dict(enumerate(self.weights)),
        )

        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)

        if show:
            plt.show()


@dataclass
class DetuningMap(WeightMap):
    """Defines a DetuningMap.

    A DetuningMap associates a detuning weight to the coordinates of a trap.
    The sum of the provided weights must be equal to 1.

    Args:
        trap_coordinates: an array containing the coordinates of the traps.
        weights: A list of detuning weights to associate to the traps.
    """
