"""Errors raised when serialization or deserializing values."""

from dataclasses import dataclass

from pulser.exceptions.base import PulserError


class SerializationError(PulserError):
    """Exception raised when sequence serialization fails."""

    pass


class SerializationSupportMissing(SerializationError):
    """Attempting to serialize a class that we don't know how to serialize."""

    pass


@dataclass
class SerializationSupportModuleMissing(SerializationSupportMissing):
    """Error: we don't know how to serialize values from this module."""

    module: str

    def __str__(self) -> str:
        return f"No serialization support for module '{self.module}'."


@dataclass
class SerializationSupportAttributeMissing(SerializationSupportMissing):
    """Error: we don't know how to serialize values from this submodule."""

    module: str
    submodule: str

    def __str__(self) -> str:
        return (
            "No serialization support for attributes of "
            f"'{self.module}.{self.submodule}'."
        )


@dataclass
class SerializationSupportClassMissing(SerializationSupportMissing):
    """Error: we don't know how to serialize values of this class."""

    module: str
    class_name: str

    def __str__(self) -> str:
        return (
            "No serialization support for "
            f"'{self.module}.{self.class_name}'."
        )


class AbstractReprError(PulserError):
    """Exception raised for abstract representation errors.

    Raised when an error occurs during the serialization to or deserialization
    from the abstract representation.
    """

    pass


class DeserializeDeviceError(PulserError):
    """Exception raised when device deserialization fails."""

    pass
