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

from collections.abc import Callable
from functools import partialmethod
from itertools import chain
import inspect
import operator
import warnings
from typing import Any, Union, TYPE_CHECKING

from pulser.json.utils import obj_to_dict
from pulser.parametrized import Parametrized

if TYPE_CHECKING:
    from pulser.parametrized import Variable  # pragma: no cover

# Available operations on parametrized objects with OpSupport
reversible_ops = [
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__pow__",
    "__mod__",
]


class OpSupport:
    """Methods for supporting operators on parametrized objects."""

    def _do_op(self, op_name: str, other: Union[int, float]) -> ParamObj:
        return ParamObj(getattr(operator, op_name), self, other)

    def _do_rop(self, op_name: str, other: Union[int, float]) -> ParamObj:
        return ParamObj(getattr(operator, op_name), other, self)

    def __neg__(self) -> ParamObj:
        return ParamObj(operator.neg, self)

    def __abs__(self) -> ParamObj:
        return ParamObj(operator.abs, self)


# Inject operator magic methods into OpSupport
for method in reversible_ops:
    rmethod = "__r" + method[2:]
    setattr(OpSupport, method, partialmethod(OpSupport._do_op, method))
    setattr(OpSupport, rmethod, partialmethod(OpSupport._do_rop, method))


class ParamObj(Parametrized, OpSupport):
    """Holds a call to a given class.

    When called, a ParamObj instance returns `cls(*args, **kwargs)`.

    Args:
        cls (callable): The object to call. Usually it's a class that's
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
            return obj_to_dict(
                self, _build=False, _name=cls.__name__, _module=cls.__module__
            )

        args = list(self.args)
        if isinstance(self.cls, Parametrized):
            cls_dict = self.cls._to_dict()
        elif hasattr(args[0], self.cls.__name__) and inspect.isfunction(
            self.cls
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

    def __getattr__(self, name: str) -> ParamObj:
        if hasattr(self.cls, name):
            return ParamObj(getattr, self, name)
        else:
            raise AttributeError(f"No attribute named '{name}' in {self}.")

    def __str__(self) -> str:
        args = [str(a) for a in self.args]
        kwargs = [f"{key}={str(value)}" for key, value in self.kwargs.items()]
        if isinstance(self.cls, Parametrized):
            name = str(self.cls)
        elif (
            hasattr(self.args[0], self.cls.__name__)
            and inspect.isfunction(self.cls)
            and inspect.isclass(self.args[0])
        ):
            name = f"{self.args[0].__name__}.{self.cls.__name__}"
            args = args[1:]
        else:
            name = self.cls.__name__
        return f"{name}({', '.join(args+kwargs)})"
