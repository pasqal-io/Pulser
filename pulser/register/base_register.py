# Copyright 2021 Pulser Development Team
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
"""Defines the abstract register class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Iterable
from collections.abc import Sequence as abcSequence
from typing import Any, cast, Optional, Union, TypeVar, Type

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from pulser.json.utils import obj_to_dict

T = TypeVar("T", bound="BaseRegister")
QubitId = Union[int, str]


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

    @abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        qs = dict(zip(self._ids, map(np.ndarray.tolist, self._coords)))
        return obj_to_dict(self, qs)

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return False

        return set(self._ids) == set(other._ids) and all(
            (
                np.array_equal(
                    self._coords[i],
                    other._coords[other._ids.index(id)],
                )
                for i, id in enumerate(self._ids)
            )
        )
