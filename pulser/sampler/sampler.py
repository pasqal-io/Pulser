"""Exposes the sample() functions.

It contains many helpers.
"""
from __future__ import annotations

import enum
import itertools
from typing import Callable, List, Optional, cast

import numpy as np

import pulser.sampler.noises as noises
from pulser.channels import Channel
from pulser.pulse import Pulse
from pulser.sampler.noises import NoiseModel
from pulser.sampler.samples import QubitSamples, Samples
from pulser.sequence import Sequence, _TimeSlot


def sample(
    seq: Sequence,
    modulation: bool = False,
    common_noises: Optional[list[NoiseModel]] = None,
    global_noises: Optional[list[NoiseModel]] = None,
) -> dict:
    """Samples the given Sequence and returns a nested dictionary.

    It is intended to be used like the json.dumps() function.

    Args:
        seq (Sequence): a pulser.Sequence instance.
        modulation (bool): a flag to account for the modulation of AOM/EOM
            before sampling.
        common_noises (Optional[list[LocalNoise]]): a list of the noise sources
            for all channels.
        global_noises (Optional[list[LocalNoise]]): a list of the noise sources
            for global channels.

    Returns:
        A nested dictionnary of the samples of the amplitude, detuning and
        phase at every nanoseconds for all channels.
    """
    if common_noises is None:
        common_noises = []
    if global_noises is None:
        global_noises = []

    # The idea of this refactor: every channel is local behind the scene A
    # global channel is just a convenient abstraction for an ideal case. But as
    # soon as we introduce noises it's useless. Hence, the distinction between
    # the two should be very thin: no big if diverging branches
    #
    # 1. determine if the global channel decay to a local one
    # 2. extract samples
    # 3. modulate
    # 4. apply noises/SLM
    # 5. write samples

    samples: dict[str, Samples | list[QubitSamples]] = {}

    # First extract to the internal representation
    for ch_name, ch in seq.declared_channels.items():
        s: Samples | list[QubitSamples]

        addr = seq.declared_channels[ch_name].addressing

        ch_noises = list(common_noises)

        if addr == "Global":
            decay = (
                len(seq._slm_mask_targets) > 0
                or len(global_noises) > 0
                or len(common_noises) > 0
            )
            if decay:
                addr = "Local"
                ch_noises.extend(global_noises)

        if addr == "Global":
            s = _sample_global_channel(seq, ch_name)
            if modulation:
                s = _modulate_global(ch, s)
            # No SLM since not decayed
            samples[ch_name] = s
            continue

        strategy = _group_between_retargets if modulation else _regular
        s = _sample_channel(seq, ch_name, strategy)
        if modulation:
            s = _modulate_local(ch, s)

        unmasked_qubits = seq._qids - seq._slm_mask_targets
        s = [x for x in s if x.qubit in unmasked_qubits]  # SLM

        s = noises.apply(s, ch_noises)

        samples[ch_name] = s

    # Output: format the samples in the simulation dict form
    d = _write_dict(seq, modulation, samples)

    return d


def _write_global_samples(d: dict, basis: str, samples: Samples) -> None:
    d["Global"][basis]["amp"] += samples.amp
    d["Global"][basis]["det"] += samples.det
    d["Global"][basis]["phase"] += samples.phase


def _write_local_samples(
    d: dict, basis: str, samples: list[QubitSamples]
) -> None:
    for s in samples:
        d["Local"][basis][s.qubit]["amp"] += s.amp
        d["Local"][basis][s.qubit]["det"] += s.det
        d["Local"][basis][s.qubit]["phase"] += s.phase


def _prepare_dict(seq: Sequence, N: int) -> dict:
    """Constructs empty dict of size N.

    Usually N is the duration of seq.
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


def _write_dict(
    seq: Sequence,
    modulation: bool,
    samples: dict[str, Samples | list[QubitSamples]],
) -> dict:
    """Needs to be rewritten: do not need the sequence nor modulation args."""
    N = seq.get_duration()
    if modulation:
        max_rt = max([ch.rise_time for ch in seq.declared_channels.values()])
        N += 2 * max_rt
    d = _prepare_dict(seq, N)

    for ch_name, a in samples.items():
        basis = seq.declared_channels[ch_name].basis
        if isinstance(a, Samples):
            _write_global_samples(d, basis, a)
        else:
            _write_local_samples(d, basis, a)
    return d


def _sample_global_channel(seq: Sequence, ch_name: str) -> Samples:
    """Compute Samples for a global channel."""
    if seq.declared_channels[ch_name].addressing != "Global":
        raise ValueError(f"{ch_name} is no a global channel")
    slots = seq._schedule[ch_name]
    return _sample_slots(seq.get_duration(), *slots)


def _sample_channel(
    seq: Sequence, ch_name: str, strategy: TimeSlotExtractionStrategy
) -> list[QubitSamples]:
    """Compute a list of QubitSamples for a channel."""
    qs: list[QubitSamples] = []
    grouped_slots = strategy(seq._schedule[ch_name])

    for group in grouped_slots:
        # Same target in one group, guaranteed by the strategy (this seems
        # weird, it's not enforced by the structure,bad design?)
        targets = group[0].targets
        ss = [
            QubitSamples.from_global(
                q, _sample_slots(seq.get_duration(), *group)
            )
            for q in targets
        ]
        qs.extend(ss)
    return qs


def _sample_slots(N: int, *slots: _TimeSlot) -> Samples:
    """Gather samples of a list of _TimeSlot in a single Samples instance.

    Args:
        N (int): the size of the samples arrays.
        *slots (tuple[_TimeSlots]): the _TimeSlots to sample

    Returns:
        A Samples instance.
    """
    samples = Samples(np.zeros(N), np.zeros(N), np.zeros(N))
    for s in slots:
        if type(s.type) is str:
            continue
        pulse = cast(Pulse, s.type)
        samples.amp[s.ti : s.tf] += pulse.amplitude.samples
        samples.det[s.ti : s.tf] += pulse.detuning.samples
        samples.phase[s.ti : s.tf] += pulse.phase

    return samples


TimeSlotExtractionStrategy = Callable[[List[_TimeSlot]], List[List[_TimeSlot]]]
"""Extraction strategy of _TimeSlot's of a Channel.

It's an alias for functions that returns a list of lists of _TimeSlots.
_TimeSlots in the same group MUST share the same targets.

NOTE:
    This strategy type is used mostly for the necessity to extract samples
    differently when taking into account the modulation of AOM/EOM. Despite
    there are only two cases, whether it's necessary to modulate a local
    channel or not, this pattern can accomodate for future needs.
"""


def _regular(ts: list[_TimeSlot]) -> list[list[_TimeSlot]]:
    """No grouping performed."""
    return [[x] for x in ts]


class _GroupType(enum.Enum):
    PULSE_AND_DELAYS = "pulses_and_delays"
    OTHER = "other"


def _key_func(x: _TimeSlot) -> _GroupType:
    if isinstance(x.type, Pulse) or x.type == "delay":
        return _GroupType.PULSE_AND_DELAYS
    else:
        return _GroupType.OTHER


def _group_between_retargets(
    ts: list[_TimeSlot],
) -> list[list[_TimeSlot]]:
    """Filter and group _TimeSlots together.

    Group the input slots by group of consecutive Pulses and delays between two
    target operations. Consider the following sequence consisting of pulses A B
    C D E F, targeting different qubits:

    .---A---B------.---C--D--E---.----F--
    ^              ^             ^
    |              |             |
    target q0   target q1     target q0

    It will group the pulses' _TimeSlot's in batches (A B), (C D E) and (F),
    returning the following list of tuples:

    [("q0", [A, B]), ("q1", [C, D, E]), ("q0", [F])]

    Returns:
        A list of tuples (a, b) where a is the list of common targeted qubits
        and b is a list of consecutive _TimeSlot of type Pulse or "delay". All
        "target" _TimeSlots are discarded.
    """
    grouped_slots: list[list[_TimeSlot]] = []

    for key, group in itertools.groupby(ts, _key_func):
        g = list(group)
        if key != _GroupType.PULSE_AND_DELAYS:
            continue
        grouped_slots.append(g)

    return grouped_slots


def _modulate_global(ch: Channel, samples: Samples) -> Samples:
    """Modulate global samples according to the hardware specs.

    Additional parameters will probably be needed (keep_end, etc).
    """
    return Samples(
        amp=ch.modulate(samples.amp),
        det=ch.modulate(samples.det),
        phase=ch.modulate(samples.phase),
    )


def _modulate_local(
    ch: Channel, samples: list[QubitSamples]
) -> list[QubitSamples]:
    """Modulate local samples according to the hardware specs.

    Additional parameters will probably be needed (keep_end, etc).
    """
    modulated_samples: list[QubitSamples] = []
    for s in samples:
        modulated_samples.append(
            QubitSamples(
                amp=ch.modulate(s.amp),
                det=ch.modulate(s.det),
                phase=ch.modulate(s.phase),
                qubit=s.qubit,
            )
        )
    return modulated_samples
