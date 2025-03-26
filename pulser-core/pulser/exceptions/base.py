"""Base exceptions raised by Pulser."""


class PulserError(Exception):
    """Any error raised by Pulser."""

    pass


class PulserValueError(ValueError, PulserError):
    """A ValueError raised by Pulser.

    Usage:
        As of this writing, most of the errors raised by Pulser are subclasses
        of PulserValueError, as we rely on them being catchable as ValueError
        for the sake of backwards compatibility.

        This is *only* for the sake of backwards compatibility. New errors
        raised by Pulser are expected to be subclasses of `PulserError` and
        will often *not* be subclasses of `PulserValueError`.
    """

    pass
