import json

import numpy as np
from pytest import mark

from pulser.backend import (
    BitStrings,
    CorrelationMatrix,
    EmulationConfig,
    Energy,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    StateResult,
)

# TODO: decide where to put these tests


@mark.parametrize(
    "observable, base_tag, kwargs",
    [
        (
            StateResult,
            "state",
            {
                "evaluation_times": [0.1, 0.3, 1.0],
            },
        ),
        (
            BitStrings,
            "bitstrings",
            {
                "evaluation_times": [i * 0.05 for i in range(10)],
                "num_shots": 211,
                "one_state": "r",
                "tag_suffix": "7",
            },
        ),
        (
            CorrelationMatrix,
            "correlation_matrix",
            {"one_state": "r"},
        ),
        (
            Occupation,
            "occupation",
            {"one_state": "g"},
        ),
        (
            Energy,
            "energy",
            {"evaluation_times": [i * 0.05 for i in range(10)]},
        ),
        (
            EnergyVariance,
            "energy_variance",
            {"evaluation_times": np.linspace(0, 1, 13)},
        ),
        (
            EnergySecondMoment,
            "energy_second_moment",
            {"evaluation_times": [i * 0.1 for i in range(5)]},
        ),
    ],
)
def test_observable_repr(observable, base_tag, kwargs):
    obs = observable(**kwargs)

    # test repr as dict
    obs_repr = obs._to_abstract_repr()
    assert isinstance(obs_repr, dict)
    assert obs_repr.keys() == {obs.tag}
    assert obs_repr[obs.tag].keys() == {obs._base_tag}

    obs_repr_kwargs = obs_repr[obs.tag][base_tag]
    for key, expected_value in kwargs.items():
        assert obs_repr_kwargs[key] is expected_value
    # test default values
    if "evaluation_times" not in kwargs.keys():
        assert obs_repr_kwargs["evaluation_times"] is None
    if "tag_suffix" not in kwargs.keys():
        assert obs_repr_kwargs["tag_suffix"] is None


def test_config_repr():
    evaluation_times = [0.1, 0.3, 0.9]
    bitstrings = BitStrings(evaluation_times=evaluation_times)
    correlation = CorrelationMatrix()
    observables = (bitstrings, correlation)
    with_modulation = True
    default_evaluation_times = "Full"
    expected_kwargs = {
        "observables": observables,
        "with_modulation": with_modulation,
        "default_evaluation_times": default_evaluation_times,
    }

    config = EmulationConfig(**expected_kwargs)
    # dump with AbstrctReprEncoder
    config_str = config.to_abstract_repr()
    # load and redump but with default JSON encoder
    # equivalent to go key by key and check single str repr
    config_load_dump_str = json.dumps(json.loads(config_str))

    assert config_str == config_load_dump_str
