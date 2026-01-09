from pulser_simulation.qutip_state import QutipState


def density_matrix_aggregator(values: list[QutipState]) -> QutipState:
    acc = values[0]._state
    if not acc.isoper:
        acc = acc * acc.dag()
    for state in values[1:]:
        if not state._state.isoper:
            q_state = state._state * state._state.dag()
        else:
            q_state = state._state
        acc += q_state
    return QutipState(acc, eigenstates=values[0].eigenstates)
