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

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import numpy as np
from scipy.spatial import KDTree


class Register:
    """A quantum register containing a set of qubits.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates (in μm) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(self, qubits):
        """Initializes a custom Register."""
        if not isinstance(qubits, dict):
            raise TypeError("The qubits have to be stored in a dictionary "
                            "matching qubit ids to position coordinates.")
        self._ids = list(qubits.keys())
        coords = [np.array(v, dtype=float) for v in qubits.values()]
        self._dim = coords[0].size
        if (any(c.shape != (self._dim,) for c in coords) or
           (self._dim != 2 and self._dim != 3)):
            raise ValueError("All coordinates must be specified as vectors of"
                             " size 2 or 3.")
        self._coords = coords

    @property
    def qubits(self):
        """Dictionary of the qubit names and their position coordinates."""
        return dict(zip(self._ids, self._coords))

    @classmethod
    def from_coordinates(cls, coords, center=True, prefix=None):
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
        """
        if center:
            coords = coords - np.mean(coords, axis=0)      # Centers the array
        if prefix is not None:
            pre = str(prefix)
            qubits = {pre+str(i): pos for i, pos in enumerate(coords)}
        else:
            qubits = dict(enumerate(coords))
        return cls(qubits)

    @classmethod
    def rectangle(cls, rows, columns, spacing=4, prefix=None):
        """Initializes the register with the qubits in a rectangular array.

        Args:
            rows (int): Number of rows.
            columns (int): Number of columns.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)
        """
        coords = np.array([(x, y) for y in range(rows)
                           for x in range(columns)], dtype=float) * spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def square(cls, side, spacing=4, prefix=None):
        """Initializes the register with the qubits in a square array.

        Args:
            side (int): Side of the square in number of qubits.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
        """
        return cls.rectangle(side, side, spacing=spacing, prefix=prefix)

    @classmethod
    def triangular_lattice(cls, rows, atoms_per_row, spacing=4, prefix=None):
        """Initializes the register with the qubits in a triangular lattice.

        Initializes the qubits in a triangular lattice pattern, more
        specifically a triangular lattice with horizontal rows, meaning the
        triangles are pointing up and down.

        Args:
            rows (int): Number of rows.
            atoms_per_row (int): Number of atoms per row.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in μm.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
        """
        coords = np.array([(x, y) for y in range(rows)
                           for x in range(atoms_per_row)], dtype=float)
        coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
        coords[:, 1] *= np.sqrt(3) / 2
        coords *= spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    def rotate(self, degrees):
        """Rotates the array around the origin by the given angle.

        Args:
            degrees (float): The angle of rotation in degrees.
        """
        if self._dim != 2:
            raise NotImplementedError("Can only rotate arrays in 2D.")
        theta = np.deg2rad(degrees)
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        self._coords = [rot @ v for v in self._coords]

    def draw(self, with_labels=True, blockade_radius=None, draw_graph=True,
             draw_half_radius=False):
        """Draws the entire register.

        Keyword args:
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

        Note:
            When drawing half the blockade radius, we say there is a blockade
            effect between atoms whenever their respective circles overlap.
            This representation is preferred over drawing the full Rydberg
            radius because it helps in seeing the interactions between atoms.
        """
        if self._dim != 2:
            raise NotImplementedError("Can only draw register layouts in 2D.")
        pos = np.array(self._coords)
        diffs = np.max(pos, axis=0) - np.min(pos, axis=0)
        diffs[diffs < 9] *= 1.5
        diffs[diffs < 9] += 2
        if blockade_radius and draw_half_radius:
            diffs[diffs < blockade_radius] = blockade_radius
        big_side = max(diffs)
        proportions = diffs / big_side
        Ls = proportions * min(big_side/4, 10)  # Figsize is, at most, (10,10)

        fig, ax = plt.subplots(figsize=Ls)
        ax.scatter(pos[:, 0], pos[:, 1], s=30, alpha=0.7, c='darkgreen')

        ax.set_xlabel("µm")
        ax.set_ylabel("µm")
        ax.axis('equal')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        if with_labels:
            for q, coords in zip(self._ids, self._coords):
                ax.annotate(q, coords, fontsize=12, ha='left', va='bottom')

        if draw_half_radius:
            if blockade_radius is None:
                raise ValueError("Define 'blockade_radius' to draw.")
            if len(pos) == 1:
                raise NotImplementedError("Needs more than one atom to draw "
                                          "the blockade radius.")

            delta_um = np.linalg.norm(pos[1] - pos[0])
            r_pts = np.linalg.norm(
                        np.subtract(*ax.transData.transform(pos[:2]).tolist())
                        ) / delta_um * blockade_radius / 2
            # A 'scatter' marker of size s has area pi/4 * s
            ax.scatter(pos[:, 0], pos[:, 1], s=4*r_pts**2, alpha=0.1,
                       c='darkgreen')
        if draw_graph and blockade_radius is not None:
            epsilon = 1e-9      # Accounts for rounding errors
            edges = KDTree(pos).query_pairs(blockade_radius * (1 + epsilon))
            lines = pos[(tuple(edges),)]
            lc = mc.LineCollection(lines, linewidths=0.6, colors='grey')
            ax.add_collection(lc)

        else:
            # Only draw central axis lines when not drawing the graph
            ax.axvline(0, c='grey', alpha=0.5, linestyle=':')
            ax.axhline(0, c='grey', alpha=0.5, linestyle=':')

        plt.show()
