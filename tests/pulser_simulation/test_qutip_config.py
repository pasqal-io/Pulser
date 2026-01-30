import re

import numpy as np
import pytest

from pulser import NoiseModel
from pulser.backend.default_observables import StateResult
from pulser_simulation.qutip_config import (
    QutipConfig,
    QutipOperator,
    QutipState,
)


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


def test_samples_per_run():
    with pytest.warns(
        UserWarning,
        match="The number of samples per run .* is ignored "
        "when using QutipBackendV2.",
    ):
        with pytest.warns(
            DeprecationWarning,
            match="Setting samples_per_run different to 1 is",
        ):
            QutipConfig(
                observables=[
                    StateResult(evaluation_times=[1.0]),
                ],
                noise_model=NoiseModel(temperature=45, samples_per_run=5),
            )


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


def test_preferred_types():
    assert QutipConfig.state_type is QutipState
    assert QutipConfig.operator_type is QutipOperator


def test_progress_bar():
    config = QutipConfig(
        observables=[
            StateResult(evaluation_times=[1.0]),
        ],
        progress_bar=True,
    )
    assert config.progress_bar
    assert "progress_bar" in config._expected_kwargs()
