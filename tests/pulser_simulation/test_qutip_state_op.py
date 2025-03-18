# Copyright 2024 Pulser Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import re

import numpy as np
import pytest
import qutip

from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.exceptions import AbstractReprError
from pulser_simulation import QutipOperator, QutipState


@pytest.fixture
def ket_r():
    return QutipState(qutip.basis(2, 0), eigenstates=("r", "g"))


@pytest.fixture
def dm_g():
    return QutipState(
        qutip.basis(2, 1) * qutip.basis(2, 1).dag(), eigenstates=("r", "g")
    )


@pytest.fixture
def ket_plus():
    return QutipState.from_state_amplitudes(
        eigenstates=("r", "g"),
        amplitudes={"r": 1 / np.sqrt(2), "g": 1 / np.sqrt(2)},
    )


class TestQutipState:

    def test_init(self):
        with pytest.raises(
            ValueError,
            match="eigenstates must be represented by single characters",
        ):
            QutipState(qutip.basis(2, 0), eigenstates=["ground", "rydberg"])
        with pytest.raises(ValueError, match="can't contain repeated entries"):
            QutipState(qutip.basis(2, 0), eigenstates=["r", "g", "r"])
        with pytest.raises(TypeError, match="must be a qutip.Qobj"):
            QutipState(np.arange(16), eigenstates=["r", "g"])
        with pytest.raises(
            TypeError, match="must be a qutip.Qobj with one of types"
        ):
            QutipState(
                qutip.operator_to_vector(qutip.ket2dm(qutip.basis(2, 0))),
                eigenstates=["r", "g"],
            )

        with pytest.raises(
            ValueError, match="incompatible with a system of 3-level qudits"
        ):
            QutipState(qutip.basis(2, 0), eigenstates=["r", "g", "h"])
        state = QutipState(
            qutip.basis(3, 0).dag(), eigenstates=["r", "g", "h"]
        )
        assert state.n_qudits == 1
        assert state.qudit_dim == 3
        assert state.eigenstates == ("r", "g", "h")
        assert state.to_qobj() == qutip.basis(3, 0)
        with pytest.raises(
            RuntimeError, match="Failed to infer the 'one state'"
        ):
            state.infer_one_state()

        three_qubit_qobj = qutip.tensor([qutip.basis(2, 1)] * 3)
        state = QutipState(three_qubit_qobj, eigenstates=("r", "g"))
        assert state.n_qudits == 3
        assert state.qudit_dim == 2
        assert state.eigenstates == ("r", "g")
        assert state.to_qobj() == three_qubit_qobj
        assert state.infer_one_state() == "r"

        two_qutrit_dm = qutip.ket2dm(qutip.tensor([qutip.basis(3, 0)] * 2))
        state = QutipState(two_qutrit_dm, eigenstates=["r", "g", "h"])
        assert state.n_qudits == 2
        assert state.qudit_dim == 3
        assert state.to_qobj() == two_qutrit_dm

    @pytest.mark.parametrize(
        "eigenstates", [("g", "r"), ("g", "h"), ("u", "d"), ("0", "1")]
    )
    def test_infer_one_state(self, eigenstates):
        assert (
            QutipState(
                qutip.basis(2, 0), eigenstates=eigenstates
            ).infer_one_state()
            == eigenstates[1]
        )

    def test_get_basis_state(self):
        n_qudits = 3
        state = QutipState.from_state_amplitudes(
            eigenstates=("r", "g", "h"), amplitudes={"g" * n_qudits: 1.0}
        )
        assert state.get_basis_state_from_index(0) == "rrr"
        assert state.get_basis_state_from_index(1) == "rrg"
        assert state.get_basis_state_from_index(2) == "rrh"
        assert state.get_basis_state_from_index(3) == "rgr"
        assert state.get_basis_state_from_index(4) == "rgg"
        assert state.get_basis_state_from_index(9) == "grr"
        assert state.get_basis_state_from_index(3**n_qudits - 1) == "hhh"

        with pytest.raises(
            ValueError, match="'index' must be a non-negative integer"
        ):
            state.get_basis_state_from_index(-1)

    def test_overlap(
        self, ket_r: QutipState, dm_g: QutipState, ket_plus: QutipState
    ):
        assert ket_r.overlap(ket_r) == 1.0
        assert dm_g.overlap(ket_r) == ket_r.overlap(dm_g) == 0.0
        assert ket_plus.overlap(ket_r) == ket_r.overlap(ket_plus)
        assert np.isclose(ket_plus.overlap(ket_r), 0.5)
        assert dm_g.overlap(ket_plus) == ket_plus.overlap(dm_g)
        assert np.isclose(dm_g.overlap(ket_plus), 0.5)

        with pytest.raises(TypeError, match="expects another 'QutipState'"):
            dm_g.overlap(ket_r.to_qobj())

        with pytest.raises(
            ValueError,
            match="Can't calculate the overlap between a state with 1 "
            "2-dimensional qudits and another with 2 3-dimensional qudits",
        ):
            ket_r.overlap(
                QutipState.from_state_amplitudes(
                    eigenstates=("r", "g", "h"), amplitudes={"rr": 1.0}
                )
            )

        err_msg = (
            "Can't calculate the overlap between states with eigenstates "
            "('r', 'g') and {}."
        )
        with pytest.raises(
            ValueError, match=re.escape(err_msg.format(("u", "d")))
        ):
            ket_r.overlap(
                QutipState(qutip.basis(2, 0), eigenstates=("u", "d"))
            )
        with pytest.raises(
            NotImplementedError, match=re.escape(err_msg.format(("g", "r")))
        ):
            ket_r.overlap(
                QutipState(qutip.basis(2, 0), eigenstates=("g", "r"))
            )

    def test_probabilities(self, ket_plus: QutipState):
        amps = {
            "rr": np.sqrt(0.5),
            "gg": 1j * np.sqrt(0.5 - 1e-12),
            "gr": 1e-6,
        }
        state = QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes=amps
        )
        probs = {
            state_str: np.abs(amp) ** 2 for state_str, amp in amps.items()
        }
        state_probs = state.probabilities(cutoff=9e-13)
        assert all(np.isclose(probs[k], state_probs[k]) for k in probs)
        probs.pop("gr")
        sum_ = sum(probs.values())
        probs = {k: v / sum_ for k, v in probs.items()}
        state_probs = state.probabilities()
        assert all(np.isclose(probs[k], state_probs[k]) for k in probs)
        assert state.infer_one_state() == "r"
        assert state.bitstring_probabilities() == {
            "11": probs["rr"],
            "00": probs["gg"],
        }
        assert state.bitstring_probabilities(one_state="g") == {
            "11": probs["gg"],
            "00": probs["rr"],
        }

        dm_plus = QutipState(
            qutip.ket2dm(ket_plus.to_qobj()), eigenstates=ket_plus.eigenstates
        )
        assert dm_plus.probabilities() == {"r": 0.5, "g": 0.5}
        assert dm_plus.bitstring_probabilities() == {"0": 0.5, "1": 0.5}

    def test_sample(self, ket_r: QutipState, dm_g: QutipState):
        shots = 2000
        assert ket_r.sample(num_shots=shots) == {"1": shots}
        assert ket_r.sample(num_shots=shots, one_state="g") == {"0": shots}
        assert ket_r.sample(num_shots=shots, p_false_pos=0.1) == {"1": shots}
        assert ket_r.sample(num_shots=shots, p_false_neg=0.1)["0"] > 0

        assert dm_g.sample(num_shots=shots) == {"0": shots}
        assert dm_g.sample(num_shots=shots, one_state="g") == {"1": shots}
        assert dm_g.sample(num_shots=shots, p_false_neg=0.1) == {"0": shots}
        assert dm_g.sample(num_shots=shots, p_false_pos=0.1)["1"] > 0

    @pytest.mark.parametrize(
        "amplitudes",
        [
            {"rrh": 1.0},
            {"rr": 0.5, "rgg": np.sqrt(0.75)},
        ],
    )
    def test_from_state_amplitudes_error(self, amplitudes):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "All basis states must be combinations of eigenstates with "
                f"the same length. Expected combinations of ('r', 'g'), each "
                f"with {len(list(amplitudes)[0])} elements."
            ),
        ):
            QutipState.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes=amplitudes
            )

    def test_from_state_amplitudes(self):
        assert QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"g": 1.0}
        ).to_qobj() == qutip.basis(2, 1)
        assert QutipState.from_state_amplitudes(
            eigenstates=("g", "r"), amplitudes={"g": 1.0}
        ).to_qobj() == qutip.basis(2, 0)
        assert QutipState.from_state_amplitudes(
            eigenstates=("r", "g", "h"), amplitudes={"g": 1.0}
        ).to_qobj() == qutip.basis(3, 1)

        r = qutip.basis(2, 0)
        g = qutip.basis(2, 1)
        assert QutipState.from_state_amplitudes(
            eigenstates=("r", "g"),
            amplitudes={"rr": -0.5j, "gr": 0.5, "rg": 0.5j, "gg": -0.5},
        ).to_qobj() == -0.5j * qutip.tensor([r, r]) + 0.5 * qutip.tensor(
            [g, r]
        ) + 0.5j * qutip.tensor(
            [r, g]
        ) - 0.5 * qutip.tensor(
            [g, g]
        )

    def test_repr(self, ket_r):
        assert repr(ket_r) == (
            "QutipState\n"
            + "-" * len("QutipState")
            + f"\nEigenstates: {ket_r.eigenstates}\n"
            + repr(ket_r.to_qobj())
        )

    def test_eq(self, ket_r, dm_g):
        assert ket_r == QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"r": 1.0}
        )
        assert dm_g != QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"g": 1.0}
        )
        assert dm_g != qutip.basis(2, 1).proj()

    def test_abstract_repr(self):
        kwargs = dict(eigenstates=("r", "g"), amplitudes={"g": 1.0})
        state = QutipState.from_state_amplitudes(**kwargs)
        assert json.dumps(state, cls=AbstractReprEncoder) == json.dumps(kwargs)

        with pytest.raises(
            AbstractReprError,
            match=re.escape(
                "Failed to serialize state of type 'QutipState' because it"
                " was not created via 'QutipState.from_state_amplitudes()'"
            ),
        ):
            json.dumps(
                QutipState(state.to_qobj(), eigenstates=state.eigenstates),
                cls=AbstractReprEncoder,
            )


class TestQutipOperator:

    def test_init(self):
        with pytest.raises(
            ValueError,
            match="eigenstates must be represented by single characters",
        ):
            QutipOperator(qutip.sigmaz(), eigenstates=["ground", "rydberg"])
        with pytest.raises(ValueError, match="can't contain repeated entries"):
            QutipOperator(qutip.sigmaz(), eigenstates=["r", "g", "r"])
        with pytest.raises(TypeError, match="must be a qutip.Qobj"):
            QutipOperator(qutip.sigmaz().full(), eigenstates=["r", "g"])
        with pytest.raises(
            TypeError, match="must be a qutip.Qobj with type 'oper'"
        ):
            QutipOperator(qutip.basis(2, 0), eigenstates=["r", "g"])
        with pytest.raises(
            ValueError, match="incompatible with a system of 3-level qudits"
        ):
            QutipOperator(qutip.sigmaz(), eigenstates=["r", "g", "h"])

        pauli_z = QutipOperator(qutip.sigmaz(), eigenstates=("r", "g"))
        assert pauli_z.eigenstates == ("r", "g")
        assert (
            pauli_z.to_qobj()
            == qutip.basis(2, 0).proj() - qutip.basis(2, 1).proj()
        )

    @pytest.fixture
    def pauli_i(self):
        return QutipOperator(qutip.identity(2), eigenstates=("r", "g"))

    @pytest.fixture
    def pauli_x(self):
        return QutipOperator(qutip.sigmax(), eigenstates=("r", "g"))

    @pytest.fixture
    def pauli_y(self):
        return QutipOperator(qutip.sigmay(), eigenstates=("r", "g"))

    @pytest.fixture
    def pauli_z(self):
        return QutipOperator(qutip.sigmaz(), eigenstates=("r", "g"))

    @pytest.mark.parametrize("op_name", ["apply_to", "expect"])
    def test_errors_on_qutip_state(self, pauli_x, op_name):
        op = getattr(pauli_x, op_name)
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"'QutipOperator.{op_name}()' expects a 'QutipState' instance"
            ),
        ):
            op(qutip.basis(2, 0))
        err_msg = (
            f"Can't apply QutipOperator.{op_name}() between a QutipOperator "
            "with eigenstates ('r', 'g') and a QutipState with {}"
        )
        with pytest.raises(
            ValueError, match=re.escape(err_msg.format(("g", "h")))
        ):
            op(QutipState(qutip.basis(2, 0), eigenstates=("g", "h")))
        with pytest.raises(
            NotImplementedError, match=re.escape(err_msg.format(("g", "r")))
        ):
            op(QutipState(qutip.basis(2, 0), eigenstates=("g", "r")))

    @pytest.mark.parametrize("op_name", ["__add__", "__matmul__"])
    def test_errors_on_qutip_operator(self, pauli_x, op_name):
        op = getattr(pauli_x, op_name)
        with pytest.raises(
            TypeError,
            match=re.escape(f"'{op_name}' expects a 'QutipOperator' instance"),
        ):
            op(ket_r)
        err_msg = (
            f"Can't apply {op_name} between a QutipOperator with eigenstates "
            "('r', 'g') and a QutipOperator with {}"
        )
        with pytest.raises(
            ValueError, match=re.escape(err_msg.format(("g", "h")))
        ):
            op(QutipOperator(qutip.basis(2, 0).proj(), eigenstates=("g", "h")))

        with pytest.raises(
            NotImplementedError, match=re.escape(err_msg.format(("g", "r")))
        ):
            op(QutipOperator(qutip.basis(2, 0).proj(), eigenstates=("g", "r")))

    def test_apply_to(self, ket_r, dm_g, pauli_x: QutipOperator):
        assert pauli_x.apply_to(ket_r) == QutipState.from_state_amplitudes(
            eigenstates=("r", "g"), amplitudes={"g": 1.0}
        )
        assert pauli_x.apply_to(dm_g) == QutipState(
            qutip.basis(2, 0).proj(), eigenstates=dm_g.eigenstates
        )

    def test_expect(
        self,
        pauli_x: QutipOperator,
        pauli_y: QutipOperator,
        pauli_z: QutipOperator,
        ket_r,
        dm_g,
        ket_plus,
    ):
        assert pauli_x.expect(ket_r) == 0.0
        assert pauli_x.expect(dm_g) == 0.0
        assert np.isclose(pauli_x.expect(ket_plus), 1.0)
        ket_minus = pauli_y.apply_to(ket_plus)
        assert np.isclose(pauli_x.expect(ket_minus), -1.0)

        assert pauli_z.expect(ket_r) == 1.0
        assert pauli_z.expect(dm_g) == -1.0
        assert np.isclose(pauli_z.expect(ket_plus), 0.0)

    def test_add(self, pauli_x, pauli_y, pauli_z):
        r = qutip.basis(2, 0)
        g = qutip.basis(2, 1)
        assert pauli_x + pauli_y == QutipOperator(
            (1 - 1j) * r * g.dag() + (1 + 1j) * g * r.dag(),
            eigenstates=pauli_x.eigenstates,
        )

        assert QutipOperator(
            qutip.identity(2), eigenstates=pauli_z.eigenstates
        ) + pauli_z == QutipOperator(
            2 * r.proj(), eigenstates=pauli_z.eigenstates
        )

    def test_rmul(self, pauli_i, pauli_z):
        assert (1 - 2j) * pauli_i == QutipOperator(
            (1 - 2j) * qutip.identity(2), eigenstates=pauli_z.eigenstates
        )
        assert 0.5 * (pauli_i + pauli_z) == QutipOperator(
            qutip.basis(2, 0).proj(), eigenstates=pauli_z.eigenstates
        )

    def test_matmul(self, pauli_i, pauli_x, pauli_y, pauli_z):
        assert (
            pauli_x @ pauli_x
            == pauli_y @ pauli_y
            == pauli_z @ pauli_z
            == pauli_i
        )
        assert pauli_x @ pauli_z == -1j * pauli_y
        assert pauli_z @ pauli_x == 1j * pauli_y

    def test_from_operator_repr(self, pauli_i):
        with pytest.raises(
            ValueError,
            match="QuditOp key must be made up of two eigenstates; instead, "
            "got 'gggg'",
        ):
            QutipOperator.from_operator_repr(
                eigenstates=("r", "g"),
                n_qudits=2,
                operations=[(1.0, [({"gggg": 1.0, "rr": -1.0}, {0})])],
            )

        with pytest.raises(
            ValueError,
            match="QuditOp key must be made up of two eigenstates; instead, "
            "got 'hh'",
        ):
            QutipOperator.from_operator_repr(
                eigenstates=("r", "g"),
                n_qudits=2,
                operations=[(1.0, [({"hh": 1.0, "rr": -1.0}, {0})])],
            )

        with pytest.raises(
            ValueError,
            match="QuditOp key must be made up of two eigenstates; instead, "
            "got 'hh'",
        ):
            QutipOperator.from_operator_repr(
                eigenstates=("r", "g"),
                n_qudits=2,
                operations=[(1.0, [({"hh": 1.0, "rr": -1.0}, {0})])],
            )

        with pytest.raises(
            ValueError, match="Got invalid indices for a system with 2 qudits"
        ):
            QutipOperator.from_operator_repr(
                eigenstates=("r", "g"),
                n_qudits=2,
                operations=[(1.0, [({"gg": 1.0, "rr": -1.0}, {3, 5, 9})])],
            )

        with pytest.raises(
            ValueError,
            match=re.escape("only indices {1} were still available"),
        ):
            QutipOperator.from_operator_repr(
                eigenstates=("r", "g"),
                n_qudits=2,
                operations=[
                    (
                        1.0,
                        [
                            ({"gg": 1.0, "rr": -1.0}, {0}),
                            ({"rg": 1.0, "rg": 1.0}, {0}),
                        ],
                    )
                ],
            )

        assert QutipOperator.from_operator_repr(
            eigenstates=("r", "g", "h"),
            n_qudits=3,
            operations=[
                (1.0, [({"rr": 1.0, "hh": -1.0}, {0}), ({"gr": -1j}, {2})])
            ],
        ) == QutipOperator(
            qutip.tensor(
                [
                    qutip.basis(3, 0).proj() - qutip.basis(3, 2).proj(),
                    qutip.identity(3),
                    -1j * qutip.basis(3, 1) * qutip.basis(3, 0).dag(),
                ]
            ),
            eigenstates=("r", "g", "h"),
        )

        assert (
            QutipOperator.from_operator_repr(
                eigenstates=("r", "g"),
                n_qudits=1,
                operations=[(1, [])],
            )
            == pauli_i
        )

        assert QutipOperator.from_operator_repr(
            eigenstates=("r", "g"),
            n_qudits=2,
            operations=[(0.5, [({"rr": 1.0, "gg": -1.0}, {0})]), (0.5, [])],
        ) == QutipOperator(
            qutip.tensor(
                [
                    qutip.basis(2, 0).proj(),
                    qutip.identity(2),
                ]
            ),
            eigenstates=("r", "g"),
        )

    def test_repr(self, pauli_z):
        assert repr(pauli_z) == (
            "QutipOperator\n"
            + "-" * len("QutipOperator")
            + f"\nEigenstates: {pauli_z.eigenstates}\n"
            + repr(pauli_z.to_qobj())
        )

    def test_eq(self, pauli_i, pauli_z, dm_g):
        g_proj = 0.5 * (pauli_i + (-1) * pauli_z)
        assert g_proj == QutipOperator(
            qutip.basis(2, 1).proj(), eigenstates=pauli_i.eigenstates
        )
        assert g_proj != dm_g

    def test_abstract_repr(self):
        kwargs = dict(
            eigenstates=("r", "g"),
            n_qudits=3,
            operations=[(0.5, [({"rr": 1.0, "gg": 1.0j}, {0})]), (0.5, [])],
        )
        op = QutipOperator.from_operator_repr(**kwargs)
        ser_ops = [
            (0.5, [({"rr": 1.0, "gg": {"real": 0.0, "imag": 1.0}}, [0])]),
            (0.5, []),
        ]
        assert json.dumps(op, cls=AbstractReprEncoder) == json.dumps(
            {**kwargs, "operations": ser_ops}
        )

        with pytest.raises(
            AbstractReprError,
            match=re.escape(
                "Failed to serialize state of type 'QutipOperator' because it"
                " was not created via 'QutipOperator.from_operator_repr()'"
            ),
        ):
            json.dumps(
                QutipOperator(op.to_qobj(), eigenstates=op.eigenstates),
                cls=AbstractReprEncoder,
            )
