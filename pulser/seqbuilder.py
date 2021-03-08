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

import copy
from collections import namedtuple
from functools import wraps
from itertools import chain

from pulser.devices import MockDevice
from pulser.parametrized import Parametrized, Variable
from pulser.sequence import Sequence

_Call = namedtuple("_Call", ['name', 'args', 'kwargs'])


def _store(func):
    """Stores a Sequence building call, potentially with Variable inputs."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if hasattr(self, "_measurement"):
            raise SystemError("The sequence has been measured, no further "
                              "changes are allowed.")
        # Check if all Parametrized inputs stem from declared variables
        for x in chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                for name, var in x.variables.items():
                    if name not in self._variables:
                        raise ValueError(f"Unknown variable '{name}'.")
                    elif self._variables[name] is not var:
                        raise ValueError(
                            f"{x} has variables that don't come from this "
                            "SequenceBuilder. Use only what's returned by "
                            "this SequenceBuilder's 'declare_variable' method"
                            " as your variables."
                            )
        func(self, *args, **kwargs)
        self._calls.append(_Call(func.__name__, args, kwargs))
    return wrapper


class SequenceBuilder:
    """A blueprint for a Sequence.

    The SequenceBuilder is designed to replicate the process of building a
    regular Sequence as closely as possible, while allowing the use of
    variable parameters. These variables have to be obtained through
    `SequenceBuilder.declare_variable()` and their values have to be
    specified as arguments of `SequenceBuilder.build()`, which returns the
    corresponding Sequence.

    The variables declared through a SequenceBuilder can be used to create
    parametrized versions of Waveform and Pulse objects, which in turn can be
    added to the SequenceBuilder. Additionally, simple arithmetic operations
    involving variables are also supported and will return parametrized objects
    that are functions of the involved variables.

    Args:
        register(Register): The atom register on which to apply the pulses.
        device(PasqalDevice): A valid device in which to execute the Sequence
            (import it from ``pulser.devices``).

    Note:
        The register, device and channel declarations do not support variable
        parameters. As such, they are the same for all Sequences built by a
        specific SequenceBuilder.
    """
    def __init__(self, register, device):
        """Initializes a new pulse sequence."""
        self._root = Sequence(register, device)
        self._calls = []    # Stores the calls that make the Sequence
        self._variables = {}

    @property
    def qubit_info(self):
        """Dictionary with the qubit's IDs and positions."""
        return self._root._register.qubits

    @property
    def declared_channels(self):
        """Channels declared in this SequenceBuilder."""
        return dict(self._root._channels)

    @property
    def declared_variables(self):
        """Variables declared in this SequenceBuilder."""
        return dict(self._variables)

    @property
    def available_channels(self):
        """Channels still available for declaration."""
        return {id: ch for id, ch in self._root._device.channels.items()
                if id not in self._root._taken_channels
                or self._root._device == MockDevice}

    def declare_variable(self, name, size=1, dtype=float):
        """Declare a new variable within this SequenceBuilder.

        Args:
            name(str): The name for the variable. Must be unique within a
                SequenceBuilder.

        Keyword Args:
            size(int=1): The number of entries stored in the variable.
            dtype(default=float): The type of the data that will be assigned
                to the variable. Must be `float`, `int` or `str`.

        Returns:
            Variable: The declared Variable instance.

        Note:
            To avoid confusion, it is recommended to store the returned
            Variable instance in a Python variable with the same name.
        """
        if name in self._variables:
            raise ValueError("Name for variable is already being used.")
        var = Variable(name, dtype, size=size)
        self._variables[name] = var
        return var

    def declare_channel(self, name, channel_id, initial_target=None):
        """Declares a new channel to the Sequence.

        Args:
            name (str): Unique name for the channel in the sequence.
            channel_id (str): How the channel is identified in the device.
                Consult ``SequenceBuilder.available_channels`` to see which
                channel ID's are still available and the associated channel's
                description.

        Keyword Args:
            initial_target (set, default=None): For 'Local' adressing channels
                only. Declares the initial target of the channel. If left as
                None, the initial target will have to be set manually as the
                first addition to this channel.
        """

        if initial_target is not None:
            qubits = initial_target
            try:
                qs = set(qubits) if not isinstance(qubits, str) else {qubits}
            except TypeError:
                qs = {qubits}

            if not qs.issubset(self._root._qids):
                raise ValueError("The initial target has to be a fixed qubit "
                                 "ID belonging to the device. Use 'target' for"
                                 " a variable initial target.")

        self._root.declare_channel(name, channel_id,
                                   initial_target=initial_target)

    @_store
    def add(self, pulse, channel, protocol='min-delay'):
        """Adds a pulse to a channel.

        Args:
            pulse (pulser.Pulse): The pulse object to add to the channel.
            channel (str): The channel's name provided when declared.

        Keyword Args:
            protocol (default='min-delay'): Stipulates how to deal with
                eventual conflicts with other channels, specifically in terms
                of having to channels act on the same target simultaneously.

                - 'min-delay'
                    Before adding the pulse, introduces the smallest
                    possible delay that avoids all exisiting conflicts.
                - 'no-delay'
                    Adds the pulse to the channel, regardless of
                    existing conflicts.
                - 'wait-for-all'
                    Before adding the pulse, adds a delay that
                    idles the channel until the end of the other channels'
                    latest pulse.
        """

        self._validate_channel(channel)
        # TODO: Restrict variables in pulse according to channel specs

        valid_protocols = ['min-delay', 'no-delay', 'wait-for-all']
        if protocol not in valid_protocols:
            raise ValueError(f"Invalid protocol '{protocol}', only accepts "
                             "protocols: " + ", ".join(valid_protocols))

    @_store
    def target(self, qubits, channel):
        """Changes the target qubit of a 'Local' channel.

        Args:
            qubits (hashable, iterable): The new target for this channel. Must
                correspond to a qubit ID in device or an iterable of qubit IDs,
                when multi-qubit adressing is possible.
            channel (str): The channel's name provided when declared. Must be
                a channel with 'Local' addressing.
         """

        self._validate_channel(channel)

        try:
            qs = set(qubits) if not isinstance(qubits, str) else {qubits}
        except TypeError:
            qs = {qubits}

        if self._root._channels[channel].addressing != 'Local':
            raise ValueError("Can only choose target of 'Local' channels.")
        elif len(qs) > self._root._channels[channel].max_targets:
            raise ValueError(
                "This channel can target at most "
                f"{self._root._channels[channel].max_targets} qubits at a time"
            )

    @_store
    def delay(self, duration, channel):
        """Idles a given channel for a specific duration.

        Args:
            duration (int): Time to delay (in multiples of 4 ns).
            channel (str): The channel's name provided when declared.
        """
        self._validate_channel(channel)

    @_store
    def measure(self, basis='ground-rydberg'):
        """Measures in a valid basis.

        Args:
            basis(str): Valid basis for measurement (consult the
                'supported_bases' attribute of the selected device for
                the available options).
        """
        available = self._root._device.supported_bases
        if basis not in available:
            raise ValueError(f"The basis '{basis}' is not supported by the "
                             "selected device. The available options are: "
                             + ", ".join(list(available)))

        self._measurement = basis

    @_store
    def phase_shift(self, phi, *targets, basis='digital'):
        r"""Shifts the phase of a qubit's reference by 'phi', for a given basis.

        This is equivalent to an :math:`R_z(\phi)` gate (i.e. a rotation of the
        target qubit's state by an angle :math:`\phi` around the z-axis of the
        Bloch sphere).

        Args:
            phi (float): The intended phase shift (in rads).
            targets: The ids of the qubits on which to apply the phase
                shift.

        Keyword Args:
            basis(str): The basis (i.e. electronic transition) to associate
                the phase shift to. Must correspond to the basis of a declared
                channel.
        """

        if basis not in self._root._phase_ref:
            raise ValueError("No declared channel targets the given 'basis'.")

    @_store
    def align(self, *channels):
        """Aligns multiple channels in time.

        Introduces delays that align the provided channels with the one that
        finished the latest, such that the next action added to any of them
        will start right after the latest channel has finished.

        Args:
            channels (str): The names of the channels to align, as given upon
                declaration.
        """

        ch_set = set(channels)
        # channels have to be a subset of the declared channels
        if not ch_set <= set(self._root._channels):
            raise ValueError("All channel names must correspond to declared"
                             " channels.")
        if len(channels) != len(ch_set):
            raise ValueError("The same channel was provided more than once.")

        if len(channels) < 2:
            raise ValueError("Needs at least two channels for alignment.")

    def build(self, **vars):
        """Builds the sequence from the programmed instructions.

        Keyword Args:
            vars: The values for all the variables declared in this
                SequenceBuilder instance, indexed by the name given upon
                declaration. Check `SequenceBuilder.declared_variables` to see
                all the variables.

        Returns:
            Sequence: The Sequence built with the given variable values.

        Example:
            # Check which variables are declared
            >>> print(seq_builder.declared_variables)
            {'x': Variable(name='x', dtype=<class 'float'>, size=1),
             'y': Variable(name='y', dtype=<class 'int'>, size=3)}
            # Build a sequence with specific values for both variables
            >>> seq = seq_builder.build(x=0.5, y=[1, 2, 3])
        """
        all_keys, given_keys = self._variables.keys(), vars.keys()
        if given_keys != all_keys:
            invalid_vars = given_keys - all_keys
            if invalid_vars:
                raise TypeError("No declared variables named: "
                                + ", ".join(invalid_vars))
            missing_vars = all_keys - given_keys
            if missing_vars:
                raise TypeError("Did not receive values for variables: "
                                + ", ".join(missing_vars))

        for name, value in vars.items():
            self._variables[name]._assign(value)

        seq = copy.deepcopy(self._root)
        for call in self._calls:
            args_ = [arg.build() if isinstance(arg, Parametrized) else arg
                     for arg in call.args]
            kwargs_ = {key: val.build() if isinstance(val, Parametrized)
                       else val for key, val in call.kwargs.items()}
            getattr(seq, call.name)(*args_, **kwargs_)

        return seq

    def __getattr__(self, name):
        if hasattr(self._root, name):
            raise AttributeError(f"'{name}' attribute of 'Sequence' is not "
                                 "available in 'SequenceBuilder'.")
        else:
            raise AttributeError(
                f"'SequenceBuilder' object has no attribute '{name}'."
                )

    def __str__(self):
        prelude = "Prelude\n-------\n" + str(self._root)
        lines = ["Stored calls\n------------"]
        for i, c in enumerate(self._calls, 1):
            args = [str(a) for a in c.args]
            kwargs = [f"{key}={str(value)}" for key, value in c.kwargs.items()]
            lines.append(f"{i}. {c.name}({', '.join(args+kwargs)})")
        return prelude + "\n\n".join(lines)

    def _validate_channel(self, channel):
        if channel not in self._root._channels:
            raise ValueError("Use the name of a declared channel.")
