from devices import pasqalobj
import numpy as np
from qutip import *

def build_init_state(N):
    """
    Builds the initial state psi_0. In this case as 'all down' except for some

    Returns
    -------

    Normalized initial state
    """
    state = [basis(2,0) for _ in range(N)]
    state[0] = basis(2,0) + 0.3 * basis(2,1)
    return tensor(state).unit()
