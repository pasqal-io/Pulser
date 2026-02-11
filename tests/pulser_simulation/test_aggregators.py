import numpy as np

from pulser_simulation import QutipState, density_matrix_aggregator


def test_density_matrix_aggregator():
    state1 = QutipState.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rgg": 1.0}
    )
    state2 = QutipState.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"grg": 1.0}
    )
    state3 = QutipState.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"ggr": 1.0}
    )

    acc = density_matrix_aggregator([state1, state2])  # vector and vector
    assert np.isclose(acc._state.norm(), 1.0)
    res1 = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert np.allclose(acc._state.full(), res1)

    acc = density_matrix_aggregator([acc, state3])  # vector and matrix
    assert np.isclose(acc._state.norm(), 1.0)
    res2 = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    assert np.allclose(acc._state.full(), res2)

    acc = density_matrix_aggregator([acc, acc])  # matrix and matrix
    assert np.isclose(acc._state.norm(), 1.0)
    assert np.allclose(acc._state.full(), res2)
