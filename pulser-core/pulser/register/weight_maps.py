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

import hashlib
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from pulser.json.utils import obj_to_dict
from pulser.register._reg_drawer import RegDrawer
from pulser.register.traps import COORD_PRECISION, Traps

if TYPE_CHECKING:
    from pulser.register.base_register import QubitId

import pulser.math as pm


@dataclass(init=False, repr=False, eq=False, frozen=True)
class WeightMap(Traps, RegDrawer):
    """Defines a generic map of weights on traps.

    Args:
        trap_coordinates: An array containing the coordinates of the traps.
        weights: A list of weights (between 0 and 1) to associate to the traps.
    """

    weights: tuple[float, ...]

    def __init__(
        self,
        trap_coordinates: ArrayLike,
        weights: typing.Sequence[float],
        slug: str | None = None,
    ) -> None:
        """Initializes a new weight map."""
        super().__init__(trap_coordinates, slug)
        if len(cast(list, trap_coordinates)) != len(weights):
            raise ValueError("Number of traps and weights don't match.")
        if not (
            np.all(np.array(weights) >= 0) and np.all(np.array(weights) <= 1)
        ):
            raise ValueError("All weights must be between 0 and 1.")
        object.__setattr__(self, "weights", tuple(weights))

    @property
    def trap_coordinates(self) -> np.ndarray:
        """The array of trap coordinates, in the order they were given."""
        return self._coords_arr.as_array(detach=True)

    @property
    def sorted_weights(self) -> np.ndarray:
        """The weights sorted to match the sorted trap coordinates."""
        sorting = self._calc_sorting_order()
        return cast(np.ndarray, np.array(self.weights)[sorting])

    def get_qubit_weight_map(
        self, qubits: Mapping[QubitId, ArrayLike]
    ) -> dict[QubitId, float]:
        """Creates a map between qubit IDs and the weight on their sites."""
        qubit_weight_map = {}
        coords_arr = self.sorted_coords
        weights_arr = self.sorted_weights
        for qid, pos in qubits.items():
            matches = np.argwhere(
                np.all(
                    np.isclose(
                        coords_arr,
                        pm.AbstractArray(pos).as_array(detach=True),
                        atol=10 ** (-COORD_PRECISION),
                    ),
                    axis=1,
                )
            )
            qubit_weight_map[qid] = float(np.sum(weights_arr[matches]))
        return qubit_weight_map

    def draw(
        self,
        labels: typing.Sequence[QubitId] | None = None,
        fig_name: str | None = None,
        kwargs_savefig: dict = {},
        custom_ax: Optional[Axes] = None,
        show: bool = True,
    ) -> None:
        """Draws the detuning map.

        Args:
            labels: If defined, writes the labels next to each site. Must have
                the same length and order like the `trap_coordinates`.
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
        pos = self.trap_coordinates
        if custom_ax is None:
            _, custom_ax = self._initialize_fig_axes(pos)

        labels_ = labels if labels is not None else list(range(len(pos)))

        super()._draw_2D(
            custom_ax,
            pos,
            labels_,
            with_labels=labels is not None,
            are_traps=True,
            dmm_qubits=dict(zip(labels_, self.weights)),
        )

        if fig_name is not None:
            plt.savefig(fig_name, **kwargs_savefig)

        if show:
            plt.show()

    @property
    def _hash_object(self) -> hashlib._Hash:
        hash_ = super()._hash_object
        # Include the weights and the type in the hash
        hash_.update(self.sorted_weights.tobytes())
        hash_.update(type(self).__name__.encode())
        return hash_

    def __repr__(self) -> str:
        return f"{type(self).__name__}_{self._safe_hash().hex()}"

    def _to_dict(self) -> dict[str, Any]:
        return obj_to_dict(
            self,
            trap_coordinates=self.trap_coordinates,
            weights=self.weights,
            slug=self.slug,
        )

    def _to_abstract_repr(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(
            traps=[
                {"weight": weight, "x": x, "y": y}
                for weight, (x, y) in zip(
                    self.sorted_weights,
                    self.sorted_coords,
                )
            ]
        )
        if self.slug is not None:
            d["slug"] = self.slug
        return d


@dataclass(init=False, repr=False, eq=False, frozen=True)
class DetuningMap(WeightMap):
    """Defines a DetuningMap.

    A DetuningMap associates a detuning weight (a value between 0 and 1)
    to the coordinates of a trap.

    Args:
        trap_coordinates: An array containing the coordinates of the traps.
        weights: A list of detuning weights (between 0 and 1) to associate
            to the traps.
    """
