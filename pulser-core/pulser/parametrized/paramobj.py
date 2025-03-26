# Copyright 2020 Pulser Development Team
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
"""Contains the ParamObj and auxiliary classes for object parametrization."""

from __future__ import annotations

import inspect
import operator
import warnings
from collections.abc import Callable
from itertools import chain
from typing import TYPE_CHECKING, Any, Union, cast

import numpy as np

import pulser.math as pm
import pulser.parametrized
from pulser.exceptions.serialization import AbstractReprError
from pulser.json.abstract_repr.serializer import abstract_repr
from pulser.json.abstract_repr.signatures import (
    BINARY_OPERATORS,
    SIGNATURES,
    UNARY_OPERATORS,
)
from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized

if TYPE_CHECKING:
    from pulser.parametrized import Variable


class OpSupport:
    """Methods for supporting operators on parametrized objects."""

    # Unary operators
    def __neg__(self) -> ParamObj:
        return ParamObj(operator.neg, self)

    def __abs__(self) -> ParamObj:
        return ParamObj(operator.abs, self)

    def __ceil__(self) -> ParamObj:
        return ParamObj(pm.ceil, self)

    def __floor__(self) -> ParamObj:
        return ParamObj(pm.floor, self)

    def __round__(self, n: int = 0) -> ParamObj:
        return cast(ParamObj, (self * 10**n).rint() / 10**n)

    def rint(self) -> ParamObj:
        """Rounds the value to the nearest int."""
        # Defined because np.round looks for 'rint'
        return ParamObj(pm.round, self)

    def sqrt(self) -> ParamObj:
        """Calculates the square root of the object."""
        return ParamObj(pm.sqrt, self)

    def exp(self) -> ParamObj:
        """Calculates the exponential of the object."""
        return ParamObj(pm.exp, self)

    def log2(self) -> ParamObj:
        """Calculates the base-2 logarithm of the object."""
        return ParamObj(pm.log2, self)

    def log(self) -> ParamObj:
        """Calculates the natural logarithm of the object."""
        return ParamObj(pm.log, self)

    def sin(self) -> ParamObj:
        """Calculates the trigonometric sine of the object."""
        return ParamObj(pm.sin, self)

    def cos(self) -> ParamObj:
        """Calculates the trigonometric cosine of the object."""
        return ParamObj(pm.cos, self)

    def tan(self) -> ParamObj:
        """Calculates the trigonometric tangent of the object."""
        return ParamObj(pm.tan, self)

    def tanh(self) -> ParamObj:
        """Calculates the hyperbolic tangent of the object."""
        return ParamObj(pm.tanh, self)

    # Binary operators
    def __add__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.add, self, other)

    def __radd__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.add, other, self)

    def __sub__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.sub, self, other)

    def __rsub__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.sub, other, self)

    def __mul__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.mul, self, other)

    def __rmul__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.mul, other, self)

    def __truediv__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.truediv, self, other)

    def __rtruediv__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.truediv, other, self)

    def __floordiv__(self, other: Union[int, float], /) -> ParamObj:
        return (self / other).__floor__()

    def __rfloordiv__(self, other: Union[int, float], /) -> ParamObj:
        return (other / self).__floor__()

    def __pow__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.pow, self, other)

    def __rpow__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.pow, other, self)

    def __mod__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.mod, self, other)

    def __rmod__(self, other: Union[int, float], /) -> ParamObj:
        return ParamObj(operator.mod, other, self)


class ParamObj(Parametrized, OpSupport):
    """Holds a call to a given class.

    When called, a ParamObj instance returns `cls(*args, **kwargs)`.

    Args:
        cls: The object to call. Usually it's a class that's
            instantiated when called.
        args: The args for calling `cls`.
        kwargs: The kwargs for calling `cls`.
    """

    def __init__(self, cls: Callable, *args: Any, **kwargs: Any) -> None:
        """Initializes a new ParamObj."""
        self.cls = cls
        self._variables: dict[str, Variable] = {}
        if isinstance(self.cls, Parametrized):
            self._variables.update(self.cls.variables)
        for x in chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                self._variables.update(x.variables)
        self.args = args
        self.kwargs = kwargs
        self._instance = None
        self._vars_state: dict[str, int] = {}

    @property
    def _default_kwargs(self) -> dict[str, Any]:
        """The default values for the object's keyword arguments."""
        cls_signature = inspect.signature(self.cls).parameters
        return {
            param: cls_signature[param].default
            for param in cls_signature
            if cls_signature[param].default != cls_signature[param].empty
        }

    @property
    def variables(self) -> dict[str, Variable]:
        """Returns all involved variables."""
        return self._variables

    def build(self) -> Any:
        """Builds the object with its variables last assigned values."""
        vars_state = {key: var._count for key, var in self._variables.items()}
        if vars_state != self._vars_state:
            self._vars_state = vars_state
            # Builds all Parametrized arguments before feeding them to cls
            args_ = [
                arg.build() if isinstance(arg, Parametrized) else arg
                for arg in self.args
            ]
            kwargs_ = {
                key: val.build() if isinstance(val, Parametrized) else val
                for key, val in self.kwargs.items()
            }
            if isinstance(self.cls, ParamObj):
                obj = self.cls.build()
            else:
                obj = self.cls
            self._instance = obj(*args_, **kwargs_)
        return self._instance

    def _to_dict(self) -> dict[str, Any]:
        def class_to_dict(cls: Callable) -> dict[str, Any]:
            module = "numpy" if isinstance(cls, np.ufunc) else cls.__module__
            return obj_to_dict(
                self, _build=False, _name=cls.__name__, _module=module
            )

        args = list(self.args)
        if isinstance(self.cls, Parametrized):
            raise ValueError(
                "Serialization of calls to parametrized objects is not "
                "supported."
            )
        elif (
            hasattr(args[0], self.cls.__name__)
            and inspect.isfunction(self.cls)
            and self.cls.__module__ != "pulser.math"
        ):
            # Check for parametrized methods
            if inspect.isclass(self.args[0]):
                # classmethod
                cls_dict = obj_to_dict(
                    self,
                    _build=False,
                    _name=self.cls.__name__,
                    _module=self.args[0].__module__,
                    _submodule=self.args[0].__name__,
                )
                args[0] = class_to_dict(self.args[0])
            else:
                raise NotImplementedError(
                    "Instance or static method "
                    "serialization is not supported."
                )
        else:
            cls_dict = class_to_dict(self.cls)

        return obj_to_dict(self, cls_dict, *args, **self.kwargs)

    def _to_abstract_repr(self) -> dict[str, Any]:
        if isinstance(self.cls, Parametrized):
            raise ValueError(
                "Serialization of calls to parametrized objects is not "
                "supported."
            )
        op_name = self.cls.__name__
        if (
            self.args  # If it is a classmethod the first arg will be the class
            and hasattr(self.args[0], op_name)
            and inspect.isfunction(self.cls)
            and not self.cls.__module__ == "pulser.math"
        ):
            # Check for parametrized methods
            if inspect.isclass(self.args[0]):
                # classmethod
                cls_name = self.args[0].__name__
                name = f"{cls_name}.{op_name}"
                signature = SIGNATURES[
                    (
                        "Pulse"
                        if cls_name == "Pulse" and op_name != "ArbitraryPhase"
                        else name
                    )
                ]
                # No existing classmethod has *args in its signature
                assert (
                    signature.var_pos is None
                ), "Unexpected signature with VAR_POSITIONAL arguments."
                all_args = {
                    **self._default_kwargs,
                    **dict(zip(signature.all_pos_args(), self.args[1:])),
                    **self.kwargs,
                }
                if name == "Pulse.ConstantAmplitude":
                    all_args["amplitude"] = abstract_repr(
                        "ConstantWaveform", 0, all_args["amplitude"]
                    )
                    return abstract_repr("Pulse", **all_args)
                elif name == "Pulse.ConstantDetuning":
                    all_args["detuning"] = abstract_repr(
                        "ConstantWaveform", 0, all_args["detuning"]
                    )
                    return abstract_repr("Pulse", **all_args)
                else:
                    return abstract_repr(name, **all_args)
            raise NotImplementedError(
                "Instance or static method serialization is not supported."
            )
        elif op_name in SIGNATURES:
            signature = SIGNATURES[op_name]
            filtered_defaults = {
                key: value
                for key, value in self._default_kwargs.items()
                if key in signature.keyword
            }
            full_kwargs = {**filtered_defaults, **self.kwargs}
            if signature.var_pos is not None:
                # No args can be given with a keyword
                return abstract_repr(op_name, *self.args, **full_kwargs)

            all_args = {
                **full_kwargs,
                **dict(zip(signature.all_pos_args(), self.args)),
            }
            if op_name == "InterpolatedWaveform" and all_args["times"] is None:
                if isinstance(
                    all_args["values"],
                    pulser.parametrized.Variable,  # Avoids circular import
                ):
                    num_values = all_args["values"].size
                else:
                    try:
                        num_values = len(all_args["values"])
                    except TypeError:
                        raise AbstractReprError(
                            "An InterpolatedWaveform with 'values' of unknown "
                            "length and unspecified 'times' can't be "
                            "serialized to the abstract representation. To "
                            "keep the same argument for 'values', provide "
                            "compatible 'times' explicitly."
                        )

                all_args["times"] = np.linspace(0, 1, num=num_values)

            return abstract_repr(op_name, **all_args)

        elif op_name in UNARY_OPERATORS:
            return dict(expression=op_name, lhs=self.args[0])

        elif op_name in BINARY_OPERATORS:
            return dict(
                expression=op_name,
                lhs=self.args[0],
                rhs=self.args[1],
            )
        else:
            raise AbstractReprError(
                f"No abstract representation for '{op_name}'."
            )

    def __call__(self, *args: Any, **kwargs: Any) -> ParamObj:
        """Returns a new ParamObj storing a call to the current ParamObj."""
        obj = ParamObj(self, *args, **kwargs)
        warnings.warn(
            "Calls to methods of parametrized objects are only "
            "executed if they serve as arguments of other "
            "parametrized objects that are themselves built. If this"
            f" is not the case, the call to {obj} will not be "
            "executed upon sequence building.",
            stacklevel=2,
        )
        return obj

    def __str__(self) -> str:
        args = [str(a) for a in self.args]
        kwargs = [f"{key}={str(value)}" for key, value in self.kwargs.items()]
        if isinstance(self.cls, Parametrized):
            name = str(self.cls)
        elif (
            self.args
            and hasattr(self.args[0], self.cls.__name__)
            and inspect.isfunction(self.cls)
            and inspect.isclass(self.args[0])
        ):
            name = f"{self.args[0].__name__}.{self.cls.__name__}"
            args = args[1:]
        else:
            name = self.cls.__name__
        return f"{name}({', '.join(args+kwargs)})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ParamObj):
            return False
        return self.args == other.args and self.kwargs == other.kwargs

    def __hash__(self) -> int:
        return id(self)
