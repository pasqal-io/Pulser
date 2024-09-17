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

import contextlib
import json
import sys

import numpy as np
import pytest

import pulser.math as pm
from pulser.json.abstract_repr.serializer import AbstractReprEncoder
from pulser.json.coders import PulserDecoder, PulserEncoder


@pytest.mark.parametrize(
    "cast_to, requires_grad",
    [(None, False), ("array", False), ("tensor", False), ("tensor", True)],
)
def test_pad(cast_to, requires_grad):
    """Explicitly tested because it's the extensively rewritten."""
    arr = [1.0, 2.0, 3.0]
    if cast_to == "array":
        arr = np.array(arr)
    elif cast_to == "tensor":
        torch = pytest.importorskip("torch")
        arr = torch.tensor(arr, requires_grad=requires_grad)

    def check_match(arr1: pm.AbstractArray, arr2):
        if requires_grad:
            assert arr1.as_tensor().requires_grad
        np.testing.assert_array_equal(
            arr1.as_array(detach=requires_grad), arr2
        )

    # "constant" mode

    check_match(
        pm.pad(arr, 2, mode="constant"), [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    )
    check_match(
        pm.pad(arr, (2, 1), mode="constant"), [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]
    )
    check_match(
        pm.pad(arr, 1, mode="constant", constant_values=-1.0),
        [-1.0, 1.0, 2.0, 3.0, -1.0],
    )
    check_match(
        pm.pad(arr, (1, 2), mode="constant", constant_values=-1.0),
        [-1.0, 1.0, 2.0, 3.0, -1.0, -1.0],
    )
    check_match(
        pm.pad(arr, (1, 2), mode="constant", constant_values=(-1.0, 4.0)),
        [-1.0, 1.0, 2.0, 3.0, 4.0, 4.0],
    )

    # "edge" mode

    check_match(
        pm.pad(arr, 2, mode="edge"), [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]
    )
    check_match(
        pm.pad(arr, (2, 1), mode="edge"), [1.0, 1.0, 1.0, 2.0, 3.0, 3.0]
    )
    check_match(pm.pad(arr, (0, 2), mode="edge"), [1.0, 2.0, 3.0, 3.0, 3.0])


class TestAbstractArray:

    @pytest.mark.parametrize("force_array", [False, True])
    def test_no_torch(self, monkeypatch, force_array):
        monkeypatch.setitem(sys.modules, "torch", None)
        pm.AbstractArray.has_torch.cache_clear()

        val = 3.2
        arr = pm.AbstractArray(val, force_array=force_array, dtype=float)
        assert not arr.is_tensor
        with pytest.raises(RuntimeError, match="`torch` is not installed"):
            arr.as_tensor()

        assert arr.size == 1
        assert arr.shape == ((1,) if force_array else ())
        assert arr.ndim == int(force_array)
        assert arr.real == 3.2
        assert arr.dtype is np.dtype(float)
        assert repr(arr) == repr(np.array(arr))
        assert arr.detach() == arr

    @pytest.mark.parametrize("force_array", [False, True])
    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_with_torch(self, force_array, requires_grad):
        pm.AbstractArray.has_torch.cache_clear()
        torch = pytest.importorskip("torch")

        t = torch.tensor(1.0, requires_grad=requires_grad)
        arr = pm.AbstractArray(t, force_array=force_array)
        assert arr.is_tensor
        assert arr.as_tensor() == t
        assert arr.as_array(detach=requires_grad) == t.detach().numpy()
        assert arr.detach() == pm.AbstractArray(t.detach())
        assert repr(arr) == repr(t[None] if force_array else t)

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_casting(self, requires_grad):
        val = 4.1
        if requires_grad:
            torch = pytest.importorskip("torch")
            val = torch.tensor(val, requires_grad=True)

        arr = pm.AbstractArray(val)
        assert int(arr) == int(val)
        assert float(arr) == float(val)
        assert bool(arr) == bool(val)

    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize("use_tensor", [False, True])
    def test_unary_ops(self, use_tensor, scalar):
        val = np.linspace(-1, 1)
        if scalar:
            val = val[13]
        if use_tensor:
            torch = pytest.importorskip("torch")
            val = torch.tensor(val)
            lib = torch
        else:
            lib = np

        arr = pm.AbstractArray(val)
        np.testing.assert_array_equal(-arr, -val)
        np.testing.assert_array_equal(abs(arr), abs(val))
        np.testing.assert_array_equal(round(arr), lib.round(val))
        np.testing.assert_array_equal(
            round(arr, 2), lib.round(val, decimals=2)
        )

    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize("use_tensor", [False, True])
    def test_comparison_ops(self, use_tensor, scalar):
        min_, max_ = -1, 1
        val = np.linspace(min_, max_, endpoint=True)
        if scalar:
            val = val[13]
        if use_tensor:
            torch = pytest.importorskip("torch")
            val = torch.tensor(val, requires_grad=True)

        arr = pm.AbstractArray(val)
        assert np.all(arr < max_ + 1e-12)
        assert np.all(arr <= max_)
        assert np.all(arr > min_ - 1e-12)
        assert np.all(arr >= min_)
        assert np.all(arr == val)
        assert np.all(arr != val * 5)

    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize("use_tensor", [False, True])
    def test_binary_ops(self, use_tensor, scalar):
        values = np.linspace(-1, 1, endpoint=True)
        if scalar:
            val = values[13]
            assert val != 0
        else:
            val = values
        if use_tensor:
            torch = pytest.importorskip("torch")
            val = torch.tensor(val)

        arr = pm.AbstractArray(val)
        # add
        np.testing.assert_array_equal(arr + 5.0, val + 5.0)
        np.testing.assert_array_equal(arr + values, val + values)
        np.testing.assert_array_equal(2.0 + arr, val + 2.0)

        # sub
        np.testing.assert_array_equal(arr - 5.0, val - 5.0)
        np.testing.assert_array_equal(arr - values, val - values)
        np.testing.assert_array_equal(2.0 - arr, 2.0 - val)

        # mul
        np.testing.assert_array_equal(arr * 5.0, val * 5.0)
        np.testing.assert_array_equal(arr * values, val * values)
        np.testing.assert_array_equal(2.0 * arr, val * 2.0)

        # truediv
        np.testing.assert_array_equal(arr / 5.0, val / 5.0)
        # Avoid zero division
        np.testing.assert_array_equal(
            arr / (values + 2.0), val / (values + 2.0)
        )
        np.testing.assert_array_equal(2.0 / arr, 2.0 / val)

        # floordiv
        np.testing.assert_array_equal(arr // 5.0, val // 5.0)
        np.testing.assert_array_equal(
            arr // (values + 2.0), val // (values + 2.0)
        )
        np.testing.assert_array_equal(2.0 // arr, 2.0 // val)

        # pow
        np.testing.assert_array_equal(arr**5.0, val**5.0)

        np.testing.assert_array_almost_equal(
            abs(arr) ** values, abs(val) ** values
        )  # rounding errors here
        np.testing.assert_array_equal(2.0**arr, 2.0**val)

        # mod
        np.testing.assert_array_equal(arr % 5.0, val % 5.0)
        np.testing.assert_array_equal(arr % values, val % values)
        np.testing.assert_array_equal(2.0 % arr, 2.0 % val)

        # matmul
        if not scalar:
            id_ = np.eye(len(arr)).tolist()
            np.testing.assert_array_almost_equal(arr @ id_, val)
            np.testing.assert_array_almost_equal(id_ @ arr, val)

    @pytest.mark.parametrize(
        "indices",
        [
            4,
            slice(None, -1),
            slice(2, 8),
            slice(9, None),
            [1, -5, 8],
            np.array([1, 2, 4]),
            np.random.random(10) > 0.5,
        ],
    )
    @pytest.mark.parametrize(
        "use_tensor, requires_grad",
        [(False, False), (True, False), (True, True)],
    )
    def test_items(self, use_tensor, requires_grad, indices):
        val = np.linspace(-1, 1, endpoint=True, num=10)
        if use_tensor:
            torch = pytest.importorskip("torch")
            val = torch.tensor(val, requires_grad=requires_grad)

        arr = pm.AbstractArray(val)

        # getitem
        assert np.all(arr[indices] == pm.AbstractArray(val[indices]))
        assert arr[indices].is_tensor == use_tensor

        # iter
        for i, item in enumerate(arr):
            assert item == val[i]
            assert isinstance(item, pm.AbstractArray)
            assert item.is_tensor == use_tensor
            if use_tensor:
                assert item.as_tensor().requires_grad == requires_grad

        # setitem
        if not requires_grad:
            arr[indices] = np.ones(len(val))[indices]
            val[indices] = 1.0
            assert np.all(arr == val)
            assert arr.is_tensor == use_tensor

            arr[indices] = np.pi
            val[indices] = np.pi
            assert np.all(arr == val)
            assert arr.is_tensor == use_tensor
        else:
            with pytest.raises(
                RuntimeError,
                match="Failed to modify a tensor that requires grad in place.",
            ):
                arr[indices] = np.ones(len(val))[indices]

        if use_tensor:
            # Check that a np.array is converted to tensor if assign a tensor
            new_val = arr.as_array(detach=True)
            arr_np = pm.AbstractArray(new_val)
            assert not arr_np.is_tensor
            arr_np[indices] = torch.zeros_like(
                val, requires_grad=requires_grad
            )[indices]
            new_val[indices] = 0.0
            assert np.all(arr_np == new_val)
            assert arr_np.is_tensor
            # The resulting tensor requires grad if the assing one did
            assert arr_np.as_tensor().requires_grad == requires_grad

    @pytest.mark.parametrize("scalar", [False, True])
    @pytest.mark.parametrize(
        "use_tensor, requires_grad",
        [(False, False), (True, False), (True, True)],
    )
    def test_serialization(self, scalar, use_tensor, requires_grad):
        values = np.linspace(-1, 1, endpoint=True)
        if scalar:
            val = values[13]
            assert val != 0
        else:
            val = values

        if use_tensor:
            torch = pytest.importorskip("torch")
            val = torch.tensor(val, requires_grad=requires_grad)

        arr = pm.AbstractArray(val)

        context = (
            pytest.raises(
                NotImplementedError,
                match="can't be serialized without losing the "
                "computational graph",
            )
            if requires_grad
            else contextlib.nullcontext()
        )

        with context:
            assert json.dumps(arr, cls=AbstractReprEncoder) == str(
                float(val) if scalar else val.tolist()
            )

        with context:
            legacy_ser = json.dumps(arr, cls=PulserEncoder)
            deserialized = json.loads(legacy_ser, cls=PulserDecoder)
            assert isinstance(deserialized, pm.AbstractArray)
            np.testing.assert_array_equal(deserialized, val)
