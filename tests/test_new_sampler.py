from random import sample
import numpy as np
import pulser.sampler.new_sampler as sampler
import pytest


def test_post_init() -> None:
    amp = np.zeros(100)
    det = np.zeros(100)
    phase = np.zeros(100)
    with pytest.raises(AssertionError):
        sampler.ChannelSamples(
            amp, det, phase, [sampler._TimeSlot(0, 0, set("q0"))]
        )
