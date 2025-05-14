"""Errors raised because a sequence is invalid."""

# Avoid circular dependencies due to type hints, part 1.
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from pulser.exceptions.base import PulserValueError

if TYPE_CHECKING:
    # Avoid circular dependencies due to type hints, part 2.
    from pulser.devices._device_datacls import BaseDevice
    from pulser.register.base_register import QubitId, RegisterLayout


@dataclass
class InvalidSequenceError(PulserValueError):
    """Attempting to define an invalid sequence."""

    device: BaseDevice


@dataclass
class DimensionError(InvalidSequenceError):
    """An error with the number of dimensions.

    Attributes:
        invalid: The invalid number of dimensions.
    """

    # Note: Ideally, all the DimensionXXXError error should be a single
    # class, but since they have different error messages, we keep them
    # separate for backwards compatibility.

    invalid: int


@dataclass
class DimensionChoiceError(DimensionError):
    """An error with the number of dimensions."""

    expected: Sequence[int]

    def __str__(self) -> str:
        return (
            f"'dimensions' must be one of {self.expected}, "
            f"not {self.invalid}."
        )


@dataclass
class DimensionTooHighError(DimensionError):
    """An error with the number of dimensions: dimensions are too high."""

    def __str__(self) -> str:
        return (
            "The device supports register layouts of at most "
            f"{self.device.dimensions} dimensions."
        )


@dataclass
class DimensionPositionsTooHighError(DimensionError):
    """An error with the number of dimensions: dimensions are too high."""

    def __str__(self) -> str:
        return (
            f"All qubit positions must be at most {self.device.dimensions}D "
            "vectors"
        )


@dataclass
class TrapsNumberError(InvalidSequenceError):
    """An error in the number of traps.

    Attributes:
        invalid: The invalid number of traps.
        layout: The invalid layout.
    """

    invalid: int
    layout: RegisterLayout


@dataclass
class TrapsNumberTooLowError(TrapsNumberError):
    """Not enough traps."""

    def __str__(self) -> str:
        return (
            "The device requires register layouts to have "
            f"at least {self.device.min_layout_traps} traps; "
            f"{self.layout!s} has only {self.invalid}."
        )


@dataclass
class TrapsNumberTooHighError(TrapsNumberError):
    """Too many traps."""

    def __str__(self) -> str:
        return (
            "The device requires register layouts to have "
            f"at most {self.device.max_layout_traps} traps; "
            f"{self.layout!s} has {self.invalid}."
        )


@dataclass
class QubitsNumberError(InvalidSequenceError):
    """An error in the number of qubits."""


@dataclass
class MinQubitNumberError(QubitsNumberError):
    """Too few qubits for the layout.

    Attributes:
        invalid: The invalid number of qubits.
        min: The minimum number of qubits.
    """

    invalid: int
    min: int

    def __str__(self) -> str:
        return (
            "Given the number of traps in the layout and the "
            "device's minimum layout filling fraction, the given"
            f" register has too few qubits ({self.invalid}). "
            "On this device, this layout must hold at least "
            f"{self.min} qubits."
        )


@dataclass
class MaxQubitNumberError(QubitsNumberError):
    """Too many qubits for the layout.

    Attributes:
        invalid: The invalid number of qubits.
        max: The maximal number of qubits.
    """

    invalid: int
    max: int

    def __str__(self) -> str:
        return (
            "Given the number of traps in the layout and the "
            "device's maximum layout filling fraction, the given"
            f" register has too many qubits ({self.invalid}). "
            "On this device, this layout can hold at most "
            f"{self.max} qubits."
        )


@dataclass
class AtomsNumberError(InvalidSequenceError):
    """An error in the number of atoms.

    Attributes:
        invalid: The invalid number of atoms.
    """

    invalid: int

    def __str__(self) -> str:
        return (
            f"The number of atoms ({self.invalid})"
            " must be less than or equal to the maximum"
            f" number of atoms supported by this device"
            f" ({self.device.max_atom_num})."
        )


@dataclass
class DistanceError(InvalidSequenceError):
    """An error in the distance between two atoms, traps, etc.

    Attributes:
        kind: The kind of objects we're checking out (e.g. atoms, traps).
        precision_exp: A value P such that precision is 1e-P µm.
        pairs: A list of offending pairs.
    """

    kind: str
    precision_exp: int
    invalid: list[tuple[QubitId, QubitId]]

    def __str__(self) -> str:
        return (
            f"The minimal distance between {self.kind} in this device "
            f"({self.device.min_atom_distance} µm) is not respected "
            f"(up to a precision of 1e{-self.precision_exp} µm) "
            f"for the pairs: {self.invalid}"
        )


@dataclass
class RadiusError(InvalidSequenceError):
    """Something is too far from the center of the device.

    Attributes:
        kind: The kind of objects we're checking out (e.g. atoms, traps).
        invalid: A list of offending atoms (or traps, etc.)
    """

    kind: str
    invalid: list[QubitId]

    def __str__(self) -> str:
        return (
            f"All {self.kind} must be at most "
            f"{self.device.max_radial_distance} μm away from the center"
            " of the array, which is not the case "
            f"for: {self.invalid}"
        )


@dataclass
class RydbergLevelError(InvalidSequenceError):
    """Invalid Rydberg Level.

    Attributes:
        invalid: The invalid value.
        min: The minimal value.
        max (optional): The maximal (inclusive) value."
    """

    invalid: int
    min: int
    max: int

    def __str__(self) -> str:
        return f"Rydberg level should be between {self.min} and {self.max}."


@dataclass
class OptimalLayoutFillingError(InvalidSequenceError):
    """Invalid optimal layout filling.

    Attributes:
        invalid: The invalid value.
    """

    invalid: float

    def __str__(self) -> str:
        return (
            "When defined, the optimal layout filling fraction "
            "must be greater than or equal to `min_layout_filling` "
            f"({self.device.min_layout_filling}) and less than or equal to "
            f"`max_layout_filling` ({self.device.max_layout_filling}), "
            f"not {self.invalid}."
        )


@dataclass
class MinimumLayoutFillingError(InvalidSequenceError):
    """Invalid minimum layout filling.

    Attributes:
        invalid: The invalid value.
    """

    invalid: float

    def __str__(self) -> str:
        return (
            "The minimum layout filling fraction must be greater than "
            "or equal to 0. and less than `max_layout_filling` "
            f"({self.device.max_layout_filling}), not {self.invalid}."
        )


@dataclass
class MaxNumberOfTrapsError(InvalidSequenceError):
    """Invalid min/max number of traps.

    Raised whenever we attempt to
    specify a min number of traps that is larger than the max number
    of traps.
    """

    def __str__(self) -> str:
        return (
            "The maximum number of layout traps "
            f"({self.device.max_layout_traps}) must be greater than "
            "or equal to the minimum number of layout traps "
            f"({self.device.min_layout_traps})."
        )
