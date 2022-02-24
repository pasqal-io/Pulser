"""Module _sequence_sampler contains functions to recover sequences' samples.

One needs samples of a sequence for emulation purposes or for the driving of an
actual QPU. This module contains allows to extract samples from a sequence in a
form of a pandas.DataFrame and a nested dictionary.

    Examples:

    seq = Sequence(...)
    samples = get_channel_dataframes()
"""
from __future__ import annotations

import enum
import functools
import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional, cast

import numpy as np

from pulser import Register
from pulser.channels import Channel
from pulser.pulse import Pulse
from pulser.sequence import QubitId, Sequence, _TimeSlot


@dataclass
class GlobalSamples:
    """Samples gather arrays of values for amplitude, detuning and phase."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray


@dataclass
class QubitSamples:
    """Gathers samples concerning a single qubit."""

    amp: np.ndarray
    det: np.ndarray
    phase: np.ndarray
    qubit: QubitId

    @classmethod
    def from_global(cls, qubit: QubitId, s: GlobalSamples) -> QubitSamples:
        """Construct a QubitSamples from a Samples instance."""
        return cls(amp=s.amp, det=s.det, phase=s.phase, qubit=qubit)


LocalNoise = Callable[[QubitSamples], QubitSamples]


# It might be better to pass a Sequence rather than a Register.


def doppler_noise(reg: Register, std_dev: float, seed: int = 0) -> LocalNoise:
    """Generate a LocalNoise modelling the Doppler effect detuning shifts."""
    rng = np.random.default_rng(seed)
    errs = rng.normal(0.0, std_dev, size=len(reg.qubit_ids))
    detunings = dict(zip(reg.qubit_ids, errs))

    def f(s: QubitSamples) -> QubitSamples:
        det = s.det
        det[np.nonzero(s.det)] += detunings[s.qubit]
        return QubitSamples(
            amp=s.amp,
            det=det,
            phase=s.phase,
            qubit=s.qubit,
        )

    return f


def amplitude_noise(
    reg: Register, waist_width: float, seed: int = 0
) -> LocalNoise:
    """Generate a LocalNoise modelling the amplitude profile of laser beams.

    The laser of a global channel has a non-constant amplitude profile in the
    register plane. It makes global channels act differently on each qubit,
    becoming local.
    """
    rng = np.random.default_rng(seed)

    def f(s: QubitSamples) -> QubitSamples:
        r = np.linalg.norm(reg.qubits[s.qubit])

        noise_amp = rng.normal(1.0, 1.0e-3)
        noise_amp *= np.exp(-((r / waist_width) ** 2))

        amp = s.amp
        amp[np.nonzero(s.amp)] *= noise_amp
        return QubitSamples(
            amp=s.amp,
            det=s.det,
            phase=s.phase,
            qubit=s.qubit,
        )

    return f


def compose_local_noises(*functions: LocalNoise) -> LocalNoise:
    """Helper to compose multiple functions."""
    if functions is None:
        return lambda x: x
    return functools.reduce(
        lambda f, g: lambda x: f(g(x)), functions, lambda x: x
    )


def sample(
    seq: Sequence,
    local_noises: Optional[list[LocalNoise]] = None,
    modulation: bool = False,
) -> dict:
    """Samples the given Sequence and returns a nested dictionary.

    Args:
        seq (Sequence): a pulser.Sequence instance.
        local_noise (Optional[list[LocalNoise]]): a list of the noise sources
            to account for.
        modulation (bool): a flag to account for the modulation of AOM/EOM
            before sampling.

    Returns:
        A nested dictionnary of the samples of the amplitude, detuning and
        phase at every nanoseconds for all channels.
    """
    if local_noises is None:
        local_noises = []

    d = _prepare_dict(seq, seq.get_duration())
    for ch_name in seq.declared_channels:
        addr = seq.declared_channels[ch_name].addressing
        basis = seq.declared_channels[ch_name].basis

        if addr == "Global":
            ch_samples = _sample_global_channel(seq, ch_name)
            d[addr][basis]["amp"] += ch_samples.amp
            d[addr][basis]["det"] += ch_samples.det
            d[addr][basis]["phase"] += ch_samples.phase
        elif addr == "Local":
            samples: list[QubitSamples] = []
            if modulation:
                # Gatering the samples of consecutive pulses with same targets.
                # These can be safely modulated.
                samples = _sample_local_channel(
                    seq, ch_name, strategy=_group_between_retarget
                )
                samples = _modulate(seq.declared_channels[ch_name], samples)
            else:
                samples = _sample_local_channel(seq, ch_name, _regular)
            for s in samples:
                if len(local_noises) > 0:
                    # The noises are applied in the reversed order of the list
                    noise_func = compose_local_noises(*local_noises)
                    s = noise_func(s)
                d[addr][basis][s.qubit]["amp"] += s.amp
                d[addr][basis][s.qubit]["det"] += s.det
                d[addr][basis][s.qubit]["phase"] += s.phase
    return d


def _prepare_dict(seq: Sequence, N: int) -> dict:
    """Constructs empty dict of size N.

    Usually N is the duration of seq, but we allow for a longer one, in case of
    modulation for example.
    """

    def new_qty_dict() -> dict:
        return {
            "amp": np.zeros(N),
            "det": np.zeros(N),
            "phase": np.zeros(N),
        }

    def new_qdict() -> dict:
        return {qubit: new_qty_dict() for qubit in seq._qids}

    if seq._in_xy:
        return {
            "Global": {"XY", new_qty_dict()},
            "Local": {"XY": new_qdict()},
        }
    else:
        return {
            "Global": {
                basis: new_qty_dict()
                for basis in ["ground-rydberg", "digital"]
            },
            "Local": {
                basis: new_qdict() for basis in ["ground-rydberg", "digital"]
            },
        }


def _sample_global_channel(seq: Sequence, ch_name: str) -> GlobalSamples:
    if ch_name not in seq.declared_channels:
        raise ValueError(f"{ch_name} is not declared in the given Sequence")
    slots = seq._schedule[ch_name]
    return _sample_slots(seq.get_duration(), *slots)


def _sample_local_channel(
    seq: Sequence, ch_name: str, strategy: TimeSlotExtractionStrategy
) -> list[QubitSamples]:
    if ch_name not in seq.declared_channels:
        raise ValueError(f"{ch_name} is not declared in the given Sequence")
    if seq.declared_channels[ch_name].addressing != "Local":
        raise ValueError(f"{ch_name} is no a local channel")

    return strategy(seq.get_duration(), seq._schedule[ch_name])


def _sample_slots(N: int, *slots: _TimeSlot) -> GlobalSamples:
    samples = GlobalSamples(np.zeros(N), np.zeros(N), np.zeros(N))
    for s in slots:
        if type(s.type) is str:
            continue
        pulse = cast(Pulse, s.type)
        samples.amp[s.ti : s.tf] += pulse.amplitude.samples
        samples.det[s.ti : s.tf] += pulse.detuning.samples
        samples.phase[s.ti : s.tf] += pulse.phase

    return samples


# This strategy type is used mostly for the necessity to extract samples
# differently when taking into account the modulation of AOM/EOM. Still it's
# nice to keep modularity here, to accomodate for future needs.
TimeSlotExtractionStrategy = Callable[
    [int, List[_TimeSlot]], List[QubitSamples]
]


def _regular(N: int, ts: list[_TimeSlot]) -> list[QubitSamples]:
    return [
        QubitSamples.from_global(q, _sample_slots(N, slot))
        for slot in ts
        for q in slot.targets
    ]


def _group_between_retarget(N: int, ts: list[_TimeSlot]) -> list[QubitSamples]:
    qs: list[QubitSamples] = []
    grouped_slots = _consecutive_slots_between_retargets(ts)
    for targets, group in grouped_slots:
        ss = [
            QubitSamples.from_global(q, _sample_slots(N, *group))
            for q in targets
        ]
        qs.extend(ss)
    return qs


class _GroupType(enum.Enum):
    PULSE_AND_DELAYS = "pulses_and_delays"
    TARGET = "target"
    OTHER = "other"


def _consecutive_slots_between_retargets(
    ts: list[_TimeSlot],
) -> list[tuple[list[QubitId], list[_TimeSlot]]]:
    """Filter and group _TimeSlots together.

    Group the input slots by group of consecutive Pulses and delays between two
    target operations.

    Returns:
        A list of tuples (a, b) where a is the list of common targeted qubits
        and b is a list of consecutive _TimeSlot of type Pulse or "delay". All
        "target" _TimeSlots are discarded.
    """
    grouped_slots: list = []

    def key_func(x: _TimeSlot) -> _GroupType:
        if isinstance(x.type, Pulse) or x.type == "delay":
            return _GroupType.PULSE_AND_DELAYS
        else:
            return _GroupType.OTHER

    for key, group in itertools.groupby(ts, key_func):
        g = list(group)
        if key != _GroupType.PULSE_AND_DELAYS:
            continue
        grouped_slots.append((g[0].targets, g))

    return grouped_slots


def _modulate(ch: Channel, samples: list[QubitSamples]) -> list[QubitSamples]:
    """Modulate samples according to the hardware specs.

    Additional parameters will probably be needed (keep_end, etc).
    """
    modulated_samples: list[QubitSamples] = []
    for s in samples:

        modulated_samples.append(
            QubitSamples(
                amp=ch.modulate(s.amp),
                det=ch.modulate(s.det),
                phase=s.phase * np.ones(len(ch.modulate(s.amp))),
                qubit=s.qubit,
            )
        )

    return modulated_samples
