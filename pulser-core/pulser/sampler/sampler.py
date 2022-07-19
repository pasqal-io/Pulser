"""New version of the sequence sampler."""
from __future__ import annotations

from typing import cast

import numpy as np

from pulser import Sequence
from pulser.pulse import Pulse

from .. import samples


def sample(seq: Sequence) -> samples.SequenceSamples:
    """Construct samples of a Sequence.

    Alternatively, use the SequenceSample constructor.
    """
    seq_samples = {
        chname: _sample_channel(chname, seq)
        for chname in seq.declared_channels
    }
    seq_samples = {
        chname: ch_schedule.get_samples()
        for chname, ch_schedule in seq._schedule.items()
    }
    addrs = {
        chname: ch.addressing for chname, ch in seq.declared_channels.items()
    }
    bases = {chname: ch.basis for chname, ch in seq.declared_channels.items()}
    return samples.SequenceSamples(
        list(seq.declared_channels.keys()),
        seq_samples,
        addrs,
        bases,
        seq.get_duration(),
    )


def _sample_channel(chname: str, seq: Sequence) -> samples.ChannelSamples:
    N = seq.get_duration()
    # Keep only pulse slots
    channel_slots = [
        s for s in seq._schedule[chname] if isinstance(s.type, Pulse)
    ]
    amp, det, phase = np.zeros(N), np.zeros(N), np.zeros(N)
    slots: list[samples._TimeSlot] = []

    for s in channel_slots:
        pulse = cast(Pulse, s.type)
        amp[s.ti : s.tf] += pulse.amplitude.samples
        det[s.ti : s.tf] += pulse.detuning.samples
        phase[s.ti : s.tf] += pulse.phase
        slots.append(samples._TimeSlot(s.ti, s.tf, s.targets))

    return samples.ChannelSamples(amp, det, phase, slots)
