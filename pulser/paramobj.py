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
from pulser.variable import Variable


class Parametrized(ABC):

    @abstractmethod
    def __call__():
        pass


class ParamObj(Parametrized):
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        args_ = [arg() if isinstance(arg, (Parametrized, Variable)) else arg
                 for arg in self.args]
        kwargs_ = {key: val() if isinstance(val, (Parametrized, Variable))
                   else val for key, val in self.kwargs.items()}
        self._instance = self.cls(*args_, **kwargs_)
        return self._instance

    def __getattr__(self, name):
        if hasattr(self.cls, name):
            return ParamObjAttr(self, name)
        else:
            AttributeError(f"No attribute named '{name}' in {self}.")


class ParamObjAttr(Parametrized):
    def __init__(self, param_obj, attr):
        if not isinstance(param_obj, ParamObj):
            return TypeError("param_obj requires a ParamObj instance.")
        self.param_obj = param_obj
        self.attr = attr

    def __call__(self):
        if hasattr(self.param_obj, "_instance"):
            return getattr(self.param_obj._instance, self.attr)
        else:
            raise AttributeError("Trying to get an attribute from an object "
                                 "that has not been initialized.")
