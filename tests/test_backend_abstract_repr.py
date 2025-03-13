import json
from pulser.backend import BitStrings
from pulser.json.abstract_repr.serializer import AbstractReprEncoder

# TODO: decide where to put these tests


def test_bitstrings_repr():
    kwargs = {
        "evaluation_times": [0.1, 0.3, 1.0],
        "num_shots": 211,
        "one_state": "r",
        "tag_suffix": "7",
    }

    obs = BitStrings(**kwargs)

    obs_repr = obs._to_abstract_repr()
    expected_repr = {"bitstrings_7": {"bitstrings": kwargs}}
    assert obs_repr == expected_repr

    obs_str = json.dumps(obs, cls=AbstractReprEncoder)
    loaded_obs_repr = json.loads(obs_str)
    assert loaded_obs_repr == expected_repr
