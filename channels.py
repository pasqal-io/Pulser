from abc import ABC, abstractmethod


class Channel(ABC):
    """Base class for an hardware channel."""

    def __init__(self, addressing, max_abs_detuning, max_amp,
                 retarget_time=None):
        """Initialize a channel with specific charactheristics.

        Args:
            addressing (str): 'local' or 'global'.
            max_abs_detuning (tuple): Maximum possible detuning (in MHz), in
            absolute value.
            max_amp(tuple): Maximum pulse amplitude (in MHz).

        Keyword Args:
            retarget_time (default=None): Time to change the target (in ns).
        """
        if addressing == 'local':
            if retarget_time is None:
                raise ValueError("Must set retarget time for local channel.")
            self.retarget_time = int(retarget_time)

        elif addressing == 'global':
            if retarget_time is not None:
                raise ValueError("Can't set retarget time for global channel.")
        else:
            raise ValueError("Addressing can only be 'global' or 'local'.")

        self.addressing = addressing

        if max_abs_detuning < 0:
            raise ValueError("Maximum absolute detuning has to be positive.")
        self.max_abs_detuning = max_abs_detuning

        if max_amp <= 0:
            raise ValueError("Maximum channel amplitude has to be positive.")
        self.max_amp = max_amp

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def basis(self):
        """The target transition at zero detuning."""
        pass

    def __repr__(self):
        s = "({}, Max Absolute Detuning: {} MHz, Max Amplitude: {} MHz"
        config = s.format(self.addressing, self.max_abs_detuning, self.max_amp)
        if self.addressing == 'local':
            config += ", Target time: {} ns".format(self.retarget_time)
        config += f", Basis: '{self.basis}'"
        return self.name + config + ")"


class Raman(Channel):
    """Raman beam channel.

    Args:
        addressing (str): 'local' or 'global'.
        max_abs_detuning (tuple): Maximum possible detuning (in MHz), in
        absolute value.
        max_amp(tuple): Maximum pulse amplitude (in MHz).
    """
    @property
    def name(self):
        return 'Raman'

    @property
    def basis(self):
        """The target transition at zero detuning."""
        return 'digital'


class Rydberg(Channel):
    """Rydberg beam  channel.

    Args:
        addressing (str): 'local' or 'global'.
        max_abs_detuning (tuple): Maximum possible detuning (in MHz), in
        absolute value.
        max_amp(tuple): Maximum pulse amplitude (in MHz).
    """
    @property
    def name(self):
        return 'Rydberg'

    @property
    def basis(self):
        """The target transition at zero detuning."""
        return 'ground-rydberg'


class MW(Channel):
    """Microwave channel.

    Args:
        addressing (str): 'local' or 'global'.
        max_abs_detuning (tuple): Maximum possible detuning (in MHz), in
        absolute value.
        max_amp(tuple): Maximum pulse amplitude (in MHz).
    """
    @property
    def name(self):
        return 'MW'

    # TODO: Define basis for this channel
