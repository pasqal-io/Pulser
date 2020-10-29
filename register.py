import numpy as np


class Register:
    """A quantum register containing a set of qubits.

    Args:
        qubits (dict): Dictionary with the qubit names as keys and their
            position coordinates (in um) as values
            (e.g. {'q0':(2, -1, 0), 'q1':(-5, 10, 0), ...}).
    """

    def __init__(self, qubits):
        if not isinstance(qubits, dict):
            raise TypeError("The qubits have to be stored in a dictionary.")
        self._qubits = qubits

    @property
    def qubits(self):
        return dict(self._qubits)

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
            coords -= np.mean(coords, axis=0)      # Centers the array
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
            spacing(float): The distance between neighbouring qubits in um.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...)
        """
        coords = np.array([(x, y) for x in range(columns)
                           for y in range(rows)], dtype=float) * spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)

    @classmethod
    def square(cls, side, spacing=4, prefix=None):
        """Initializes the register with the qubits in a square array.

        Args:
            side (int): Side of the square in number of qubits.

        Keyword args:
            spacing(float): The distance between neighbouring qubits in um.
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
            spacing(float): The distance between neighbouring qubits in um.
            prefix (str): The prefix for the qubit ids. If defined, each qubit
                id starts with the prefix, followed by an int from 0 to N-1
                (e.g. prefix='q' -> IDs: 'q0', 'q1', 'q2', ...).
        """
        coords = np.array([(x, y) for x in range(atoms_per_row)
                           for y in range(rows)], dtype=float)
        coords[:, 0] += 0.5 * np.mod(coords[:, 1], 2)
        coords[:, 1] *= np.sqrt(3) / 2
        coords *= spacing

        return cls.from_coordinates(coords, center=True, prefix=prefix)
