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

from collections.abc import Sequence as abcSequence
from itertools import combinations
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc
from scipy.spatial import KDTree

from pulser.register.base_register import QubitId


class RegDrawer:
    """Helper functions for Register drawing."""

    @staticmethod
    def _draw_2D(
        ax: plt.axes._subplots.AxesSubplot,
        pos: np.ndarray,
        ids: abcSequence[QubitId],
        plane: tuple = (0, 1),
        with_labels: bool = True,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        masked_qubits: set[QubitId] = set(),
        are_traps: bool = False,
    ) -> None:
        ix, iy = plane

        if are_traps:
            params = dict(s=50, edgecolors="black", facecolors="none")
        else:
            params = dict(s=30, c="darkgreen")

        ax.scatter(pos[:, ix], pos[:, iy], alpha=0.7, **params)

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
            plot_ids: list[list[str]] = [[f"{i}"] for i in ids]
            # Threshold distance between points
            epsilon = 1.0e-2 * np.diff(ax.get_xlim())[0]

            i = 0
            bbs = {}
            final_plot_ids: list[str] = []
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
                final_plot_ids.append(", ".join(plot_ids[i]))
                bbs[final_plot_ids[i]] = overlap
                i += 1

            for q, coords in zip(final_plot_ids, plot_pos):
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
                    fontsize=12,
                    multialignment="right",
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
    def _draw_3D(
        pos: np.ndarray,
        ids: abcSequence[QubitId],
        projection: bool = False,
        with_labels: bool = True,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
        are_traps: bool = False,
    ) -> None:
        if draw_graph and blockade_radius is not None:
            epsilon = 1e-9  # Accounts for rounding errors
            edges = KDTree(pos).query_pairs(blockade_radius * (1 + epsilon))

        if projection:
            labels = "xyz"
            fig, axes = RegDrawer._initialize_fig_axes_projection(
                pos,
                blockade_radius=blockade_radius,
                draw_half_radius=draw_half_radius,
            )
            fig.tight_layout(w_pad=6.5)

            for ax, (ix, iy) in zip(axes, combinations(np.arange(3), 2)):
                RegDrawer._draw_2D(
                    ax,
                    pos,
                    ids,
                    plane=(
                        ix,
                        iy,
                    ),
                    with_labels=with_labels,
                    blockade_radius=blockade_radius,
                    draw_graph=draw_graph,
                    draw_half_radius=draw_half_radius,
                    are_traps=are_traps,
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

            if are_traps:
                params = dict(s=50, c="white", edgecolors="black")
            else:
                params = dict(s=30, c="darkgreen")

            for i in range(1, 3):
                ax = fig.add_subplot(
                    1, 2, i, projection="3d", azim=-60 * (-1) ** i, elev=15
                )

                ax.scatter(
                    pos[:, 0], pos[:, 1], pos[:, 2], alpha=0.7, **params
                )

                if with_labels:
                    for q, coords in zip(ids, pos):
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
                    mesh_num = 20 if len(ids) > 10 else 40
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

    @staticmethod
    def _initialize_fig_axes(
        pos: np.ndarray,
        blockade_radius: Optional[float] = None,
        draw_half_radius: bool = False,
    ) -> tuple[plt.figure.Figure, plt.axes.Axes]:
        """Creates the Figure and Axes for drawing the register."""
        diffs = RegDrawer._register_dims(
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

    @staticmethod
    def _initialize_fig_axes_projection(
        pos: np.ndarray,
        blockade_radius: Optional[float] = None,
        draw_half_radius: bool = False,
    ) -> tuple[plt.figure.Figure, plt.axes.Axes]:
        """Creates the Figure and Axes for drawing the register projections."""
        diffs = RegDrawer._register_dims(
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

    @staticmethod
    def _draw_checks(
        n_atoms: int,
        blockade_radius: Optional[float] = None,
        draw_graph: bool = True,
        draw_half_radius: bool = False,
    ) -> None:
        """Checks common in all register drawings.

        Args:
            n_atoms(int): Number of atoms in the register.
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
            if n_atoms < 2:
                raise NotImplementedError(
                    "Needs more than one atom to draw the blockade radius."
                )
