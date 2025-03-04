import re

import numpy as np
import pytest

from pulser.backend.default_observables import StateResult
from pulser_simulation.qutip_config import QutipConfig


def test_no_interaction_matrix():
    with pytest.raises(
        NotImplementedError,
        match="'QutipBackendV2' does not handle custom interaction matrices.",
    ):
        QutipConfig(
            observables=[
                StateResult(evaluation_times=[1.0]),
            ],
            interaction_matrix=np.eye(4),
        )


def test_sampling_rate():
    with pytest.raises(
        ValueError, match="be greater than 0 and less than or equal to 1"
    ):
        QutipConfig(
            observables=[
                StateResult(evaluation_times=[1.0]),
            ],
            sampling_rate=1.2,
        )

    config = QutipConfig(
        observables=[
            StateResult(evaluation_times=[1.0]),
        ],
        sampling_rate=0.5,
    )

    assert "sampling_rate" in config._expected_kwargs()


def test_initial_state():
    with pytest.raises(
        TypeError,
        match=re.escape(
            "If provided, `initial_state` must be an instance of `QutipState`"
        ),
    ):
        QutipConfig(
            observables=[
                StateResult(evaluation_times=[1.0]),
            ],
            initial_state="all-ground",
        )
