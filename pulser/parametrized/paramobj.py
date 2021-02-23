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

from functools import partialmethod
from itertools import chain
import operator

from pulser.parametrized import Parametrized

# Availabe operations on parameterized objects with OpSupport
reversible_ops = [
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__pow__",
    "__mod__"
]


class OpSupport:
    """Methods for supporting operators on parametrized objects."""

    def _do_op(self, op_name, other):
        return ParamObj(getattr(operator, op_name), self, other)

    def _do_rop(self, op_name, other):
        return ParamObj(getattr(operator, op_name), other, self)

    def __neg__(self):
        return ParamObj(operator.neg, self)

    def __abs__(self):
        return ParamObj(operator.abs, self)


# Inject operator magic methods into OpSupport
for method in reversible_ops:
    rmethod = "__r" + method[2:]
    setattr(OpSupport, method, partialmethod(OpSupport._do_op, method))
    setattr(OpSupport, rmethod, partialmethod(OpSupport._do_rop, method))


class ParamObj(Parametrized, OpSupport):
    def __init__(self, cls, *args, **kwargs):
        """Holds a call to a given class.

        When called, a ParamObj instance returns `cls(*args, **kwargs)`.

        Args:
            cls (callable): The object to call. Usually it's a class that's
                instantiated when called.
            args: The args for calling `cls`.
            kwargs: The kwargs for calling `cls`.
        """
        self.cls = cls
        self._variables = {}
        for x in chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                self._variables.update(x.variables)
        self.args = args
        self.kwargs = kwargs
        self._instance = None

    @property
    def variables(self):
        return self._variables

    def __call__(self):
        # Builds all Parametrized arguments before feeding them cls
        args_ = [arg() if isinstance(arg, Parametrized) else arg
                 for arg in self.args]
        kwargs_ = {key: val() if isinstance(val, Parametrized)
                   else val for key, val in self.kwargs.items()}
        self._instance = self.cls(*args_, **kwargs_)
        return self._instance

    def __getattr__(self, name):
        if hasattr(self.cls, name):
            return _ParamObjAttr(self, name)
        else:
            AttributeError(f"No attribute named '{name}' in {self}.")

    def __str__(self):
        args = [str(a) for a in self.args]
        kwargs = [f"{key}={str(value)}" for key, value in self.kwargs.items()]
        return f"{self.cls.__name__}({', '.join(args+kwargs)})"


class _ParamObjAttr(Parametrized, OpSupport):
    """Stores the access to the attribute of a ParamObj's build."""
    def __init__(self, param_obj, attr):
        if not isinstance(param_obj, ParamObj):
            return TypeError("ParamObjAttr requires a ParamObj instance.")
        self.param_obj = param_obj
        self.attr = attr

    @property
    def variables(self):
        return self.param_obj.variables

    def __call__(self):
        if isinstance(self.param_obj._instance, self.param_obj.cls):
            return getattr(self.param_obj._instance, self.attr)
        else:
            return getattr(self.param_obj(), self.attr)

    def __str__(self):
        return f"{str(self.param_obj)}.{self.attr}"
