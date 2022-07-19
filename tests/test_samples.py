import numpy as np
import pytest

from pulser.sampler import sampler


def test_post_init() -> None:
    amp = np.zeros(100)
    det = np.zeros(100)
    phase = np.zeros(100)
    with pytest.raises(AssertionError):
        sampler.ChannelSamples(
            amp, det, phase, [sampler._TimeSlot(0, 0, set("q0"))]
        )
