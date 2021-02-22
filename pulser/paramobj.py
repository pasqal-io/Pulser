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

from abc import ABC, abstractmethod
from itertools import chain


class Parametrized(ABC):

    @property
    @abstractmethod
    def variables(self):
        pass

    @abstractmethod
    def __call__():
        pass


class ParamObj(Parametrized):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self._variables = {}
        for x in chain(args, kwargs.values()):
            if isinstance(x, Parametrized):
                self._variables.update(x.variables)
        self.args = args
        self.kwargs = kwargs

    @property
    def variables(self):
        return self._variables

    def __call__(self):
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


class _ParamObjAttr(Parametrized):
    def __init__(self, param_obj, attr):
        if not isinstance(param_obj, ParamObj):
            return TypeError("ParamObjAttr requires a ParamObj instance.")
        self.param_obj = param_obj
        self.attr = attr

    @property
    def variables(self):
        return self.param_obj.variables

    def __call__(self):
        if hasattr(self.param_obj, "_instance"):
            return getattr(self.param_obj._instance, self.attr)
        else:
            raise AttributeError("Trying to get an attribute from an object "
                                 "that has not been initialized.")

    def __str__(self):
        return f"{str(self.param_obj)}.{self.attr}"
