import importlib
from typing import Any, Callable

# COMP_BACKEND = "torch" if importlib.util.find_spec("torch") else "numpy"

COMP_BACKEND = "numpy"


def set_comp_backend(comp_backend: str = "numpy") -> None:
    global COMP_BACKEND
    if comp_backend == "numpy":
        COMP_BACKEND = "numpy"
    elif comp_backend == "torch":
        COMP_BACKEND = "torch"
    else:
        raise ValueError(f"{comp_backend} is not suported")


class CompBackend:
    @classmethod
    @property
    def ufunc(self) -> Callable:
        lib = importlib.import_module("numpy")
        return lib.ufunc

    @classmethod
    @property
    def base_repr(self) -> Callable:
        lib = importlib.import_module("numpy")
        return lib.base_repr

    @classmethod
    @property
    def binary_repr(self) -> Callable:
        lib = importlib.import_module("numpy")
        return lib.binary_repr

    @classmethod
    @property
    def union1d(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            def _union1d(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
                res = np.union1d(t1.detach().numpy(), t2.detach().numpy())
                return torch.tensor(res)

            return _union1d
        elif COMP_BACKEND == "numpy":
            return np.union1d

    @classmethod
    @property
    def ptp(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            def _ptp(t: torch.Tensor, axis: int | None = None) -> torch.Tensor:
                res = np.ptp(t.detach().numpy(), axis=axis)
                return torch.tensor(res)

            return _ptp
        elif COMP_BACKEND == "numpy":
            return np.ptp

    @classmethod
    @property
    def insert(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            def _insert(
                t: torch.Tensor, axis: int | None = None
            ) -> torch.Tensor:
                res = np.insert(t.detach().numpy(), axis=axis)
                return torch.tensor(res)

            return _insert
        elif COMP_BACKEND == "numpy":
            return np.insert

    @classmethod
    @property
    def meshgrid(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            def _meshgrid(*t: torch.Tensor) -> list[torch.Tensor]:
                res = np.meshgrid(*t)
                return [torch.tensor(r) for r in res]

            return _meshgrid
        elif COMP_BACKEND == "numpy":
            return np.meshgrid

    @classmethod
    @property
    def mgrid(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            class _:
                def __getitem__(self, key: Any) -> torch.Tensor:
                    return torch.tensor(np.mgrid[key])

            _mgrid = _()
            return _mgrid
        elif COMP_BACKEND == "numpy":
            return np.mgrid

    @classmethod
    @property
    def lexsort(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            def _lexsort(keys: list | tuple, axis: int = -1) -> torch.Tensor:
                res = np.lexsort(keys, axis)
                return torch.tensor(res)

            return _lexsort
        elif COMP_BACKEND == "numpy":
            return np.lexsort

    @classmethod
    @property
    def searchsorted(self) -> Callable:
        np = importlib.import_module("numpy")
        if COMP_BACKEND == "torch":
            torch = importlib.import_module("torch")

            def _lexsort(
                a: torch.Tensor, v: torch.Tensor, side: str | None = None
            ) -> torch.Tensor:
                res = np.searchsorted(a, v, side)
                return torch.tensor(res)

            return _lexsort
        elif COMP_BACKEND == "numpy":
            return np.searchsorted

    @classmethod
    def load_lib(self):
        if COMP_BACKEND == "torch":
            lib = importlib.import_module("torch")
        elif COMP_BACKEND == "numpy":
            lib = importlib.import_module("numpy")
        return lib

    @classmethod
    @property
    def ndarray(self) -> Any:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.Tensor
        elif COMP_BACKEND == "numpy":
            return lib.ndarray

    @classmethod
    @property
    def array(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _array(t: list | lib.Tensor, dtype: None = None) -> lib.Tensor:
                if isinstance(t, (list, tuple)):
                    t = lib.stack([lib.as_tensor(el, dtype=dtype) for el in t])
                else:
                    t = lib.as_tensor(t, dtype=dtype)
                return t

            return _array
        elif COMP_BACKEND == "numpy":
            return lib.array

    @classmethod
    @property
    def repeat(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _repeat(
                t: lib.Tensor, repeats: int, axis: int | None = None
            ) -> lib.Tensor:
                return lib.repeat_interleave(t, repeats, dim=axis)

            return _repeat
        elif COMP_BACKEND == "numpy":
            return lib.repeat

    @classmethod
    @property
    def fftfreq(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.fft.fftfreq
        elif COMP_BACKEND == "numpy":
            from scipy.fft import fftfreq

            return fftfreq

    @classmethod
    @property
    def fft(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.fft.fft
        elif COMP_BACKEND == "numpy":
            from scipy.fft import fft

            return fft

    @classmethod
    @property
    def ifft(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.fft.ifft
        elif COMP_BACKEND == "numpy":
            from scipy.fft import ifft

            return ifft

    @classmethod
    @property
    def pdist(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.nn.functional.pdist
        elif COMP_BACKEND == "numpy":
            from scipy.spatial.distance import pdist

            return pdist

    @classmethod
    @property
    def norm(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _norm(t: lib.Tensor, axis: int = 0) -> lib.Tensor:
                return lib.linalg.norm(
                    lib.as_tensor(t, dtype=lib.float64), dim=axis
                )

            return _norm
        elif COMP_BACKEND == "numpy":
            return lib.linalg.norm

    @classmethod
    @property
    def ceil(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _ceil(t: lib.Tensor) -> lib.Tensor:
                return lib.ceil(lib.as_tensor(t, dtype=lib.float64))

            return _ceil
        elif COMP_BACKEND == "numpy":
            return lib.ceil

    @classmethod
    @property
    def all(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _all(t: lib.Tensor, axis: int = 0) -> lib.Tensor:
                return lib.all(lib.as_tensor(t, dtype=lib.float64), dim=axis)

            return _all
        elif COMP_BACKEND == "numpy":
            return lib.all
        
    @classmethod
    @property
    def log10(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _log10(t: lib.Tensor) -> lib.Tensor:
                return lib.log10(lib.as_tensor(t, dtype=lib.float64))

            return _log10
        elif COMP_BACKEND == "numpy":
            return lib.log10
        
    @classmethod
    @property
    def nonzero(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _nonzero(t: lib.Tensor) -> lib.Tensor:
                return lib.nonzero(lib.as_tensor(t, dtype=lib.float64), as_tuple=True)

            return _nonzero
        elif COMP_BACKEND == "numpy":
            return lib.nonzero

    @classmethod
    @property
    def transpose(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _transpose(t: lib.Tensor) -> lib.Tensor:
                return lib.transpose(
                    lib.as_tensor(t, dtype=lib.float64), -1, 0
                )

            return _transpose
        elif COMP_BACKEND == "numpy":
            return lib.transpose

    @classmethod
    @property
    def integer(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.IntType
        elif COMP_BACKEND == "numpy":
            return lib.integer

    @classmethod
    @property
    def blackman(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.blackman_window
        elif COMP_BACKEND == "numpy":
            return lib.blackman

    @classmethod
    @property
    def kaiser(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lambda n, beta: lib.kaiser_window(
                n, periodic=False, beta=beta
            )
        elif COMP_BACKEND == "numpy":
            return lib.kaiser

    @classmethod
    @property
    def copy(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _copy(t: lib.Tensor) -> lib.Tensor:
                return lib.clone(lib.as_tensor(t))

            return _copy
        elif COMP_BACKEND == "numpy":
            return lib.copy

    @classmethod
    @property
    def mod(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.remainder
        elif COMP_BACKEND == "numpy":
            return lib.mod

    @classmethod
    @property
    def multinomial(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _multinomial(n: int, t: lib.Tensor) -> lib.Tensor:
                out = lib.multinomial(lib.as_tensor(t), n, replacement=True).unique(return_counts=True)
                distr = lib.zeros(len(t), dtype=int)
                distr[out[0]] = out[1]
                return distr

            return _multinomial
        elif COMP_BACKEND == "numpy":
            return lib.random.multinomial

    @classmethod
    @property
    def normal(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.normal
        elif COMP_BACKEND == "numpy":
            return lib.random.normal

    @classmethod
    @property
    def uniform(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.rand
        elif COMP_BACKEND == "numpy":
            return lib.random.uniform

    @classmethod
    @property
    def rint(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":
            return lib.round
        elif COMP_BACKEND == "numpy":
            return lib.rint

    @classmethod
    @property
    def sqrt(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _sqrt(t: lib.Tensor) -> lib.Tensor:
                return lib.sqrt(lib.as_tensor(t))

            return _sqrt
        elif COMP_BACKEND == "numpy":
            return lib.sqrt

    @classmethod
    @property
    def min(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _min(t: lib.Tensor, initial: int | None = None) -> lib.Tensor:
                if initial is not None:
                    return min(lib.min(lib.as_tensor(t)), initial)
                else:
                    return lib.min(lib.as_tensor(t))

            return _min
        elif COMP_BACKEND == "numpy":
            return lib.min

    @classmethod
    @property
    def max(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _max(t: lib.Tensor) -> lib.Tensor:
                return lib.max(lib.as_tensor(t))

            return _max
        elif COMP_BACKEND == "numpy":
            return lib.max

    @classmethod
    @property
    def cos(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _cos(t: lib.Tensor) -> lib.Tensor:
                return lib.cos(lib.as_tensor(t))

            return _cos
        elif COMP_BACKEND == "numpy":
            return lib.cos
        
    @classmethod
    @property
    def deg2rad(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _deg2rad(t: lib.Tensor) -> lib.Tensor:
                return lib.deg2rad(lib.as_tensor(t))

            return _deg2rad
        elif COMP_BACKEND == "numpy":
            return lib.deg2rad
        
    @classmethod
    @property
    def sign(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _sign(t: lib.Tensor) -> lib.Tensor:
                return lib.sign(lib.as_tensor(t))

            return _sign
        elif COMP_BACKEND == "numpy":
            return lib.sign
        
    @classmethod
    @property
    def diff(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _diff(t: lib.Tensor) -> lib.Tensor:
                return lib.diff(lib.as_tensor(t))

            return _diff
        elif COMP_BACKEND == "numpy":
            return lib.diff
        
    @classmethod
    @property
    def log(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _log(t: lib.Tensor) -> lib.Tensor:
                return lib.log(lib.as_tensor(t))

            return _log
        elif COMP_BACKEND == "numpy":
            return lib.log

    @classmethod
    @property
    def sin(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _sin(t: lib.Tensor) -> lib.Tensor:
                return lib.sin(lib.as_tensor(t))

            return _sin
        elif COMP_BACKEND == "numpy":
            return lib.sin

    @classmethod
    @property
    def triu(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _triu(t: lib.Tensor, k: int = 0) -> lib.Tensor:
                t = lib.as_tensor(t)
                if len(t.shape) < 2:
                    t = t.unsqueeze(1).repeat((1, len(t)))
                return lib.triu(t, diagonal=k)

            return _triu
        elif COMP_BACKEND == "numpy":
            return lib.triu

    @classmethod
    @property
    def linspace(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _linspace(
                start: int | float, stop: int | float, num: int
            ) -> lib.Tensor:
                return lib.linspace(start, stop, steps=num)

            return _linspace
        elif COMP_BACKEND == "numpy":
            return lib.linspace

    @classmethod
    @property
    def unique(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _unique(t: lib.Tensor, axis: int | None = None) -> lib.Tensor:
                return lib.unique(lib.as_tensor(t), dim=axis)

            return _unique
        elif COMP_BACKEND == "numpy":

            def _unique(t: lib.array, axis: int | None = None) -> lib.array:
                return lib.unique(t, axis=axis)

            return _unique

    @classmethod
    @property
    def pad(self) -> Callable:
        lib = self.load_lib()
        if COMP_BACKEND == "torch":

            def _pad(
                t: lib.Tensor,
                pad_width: tuple | int,
                mode: str = "constant",
                constant_values: tuple | int | float = 0,
            ) -> lib.Tensor:
                if mode == "constant":
                    if isinstance(pad_width, int) and isinstance(
                        constant_values, (int, float)
                    ):
                        out = lib.nn.functional.pad(
                            t,
                            (pad_width, pad_width),
                            "constant",
                            constant_values,
                        )
                    elif isinstance(pad_width, tuple) and isinstance(
                        constant_values, (int, float)
                    ):
                        out = lib.nn.functional.pad(
                            t, pad_width, "constant", constant_values
                        )
                    elif isinstance(pad_width, int) and isinstance(
                        constant_values, tuple
                    ):
                        out = lib.nn.functional.pad(
                            t, (pad_width, 0), "constant", constant_values[0]
                        )
                        out = lib.nn.functional.pad(
                            out, (0, pad_width), "constant", constant_values[1]
                        )
                    else:
                        out = lib.nn.functional.pad(
                            t,
                            (pad_width[0], 0),
                            "constant",
                            constant_values[0],
                        )
                        out = lib.nn.functional.pad(
                            out,
                            (0, pad_width[1]),
                            "constant",
                            constant_values[1],
                        )
                elif mode == "edge":
                    if isinstance(pad_width, (int, float)):
                        out = lib.nn.functional.pad(
                            t, (pad_width, 0), "constant", t[0]
                        )
                        out = lib.nn.functional.pad(
                            out, (0, pad_width), "constant", t[-1]
                        )
                    else:
                        out = lib.nn.functional.pad(
                            t, (pad_width[0], 0), "constant", t[0]
                        )
                        out = lib.nn.functional.pad(
                            out, (0, pad_width[1]), "constant", t[-1]
                        )
                return out

            return _pad
        elif COMP_BACKEND == "numpy":

            def _pad(
                t: lib.array,
                pad_width: tuple | int,
                mode: str = "constant",
                constant_values: tuple | int | float = 0,
            ) -> lib.array:
                if mode == "constant":
                    return lib.pad(
                        t, pad_width, mode, constant_values=constant_values
                    )
                elif mode == "edge":
                    return lib.pad(t, pad_width, mode)

            return _pad

    @classmethod
    @property
    def ravel(self) -> Callable:
        lib = self.load_lib()
        return lib.ravel

    @classmethod
    @property
    def int8(self) -> Callable:
        lib = self.load_lib()
        return lib.int8

    @classmethod
    @property
    def prod(self) -> Callable:
        lib = self.load_lib()
        return lib.prod

    @classmethod
    @property
    def argmax(self) -> Callable:
        lib = self.load_lib()
        return lib.argmax

    @classmethod
    @property
    def dot(self) -> Callable:
        lib = self.load_lib()
        return lib.dot

    @classmethod
    @property
    def svd(self) -> Callable:
        lib = self.load_lib()
        return lib.linalg.svd

    @classmethod
    @property
    def logical_not(self) -> Callable:
        lib = self.load_lib()
        return lib.logical_not

    @classmethod
    @property
    def angle(self) -> Callable:
        lib = self.load_lib()
        return lib.angle

    @classmethod
    @property
    def where(self) -> Callable:
        lib = self.load_lib()
        return lib.where

    @classmethod
    @property
    def double(self) -> Callable:
        lib = self.load_lib()
        return lib.double

    @classmethod
    @property
    def any(self) -> Callable:
        lib = self.load_lib()
        return lib.any

    @classmethod
    @property
    def eye(self) -> Callable:
        lib = self.load_lib()
        return lib.eye

    @classmethod
    @property
    def zeros(self) -> Callable:
        lib = self.load_lib()
        return lib.zeros

    @classmethod
    @property
    def zeros_like(self) -> Callable:
        lib = self.load_lib()
        return lib.zeros_like

    @classmethod
    @property
    def round(self) -> Callable:
        lib = self.load_lib()
        return lib.round

    @classmethod
    @property
    def abs(self) -> Callable:
        lib = self.load_lib()
        return lib.abs

    @classmethod
    @property
    def mean(self) -> Callable:
        lib = self.load_lib()
        return lib.mean

    @classmethod
    @property
    def exp(self) -> Callable:
        lib = self.load_lib()
        return lib.exp

    @classmethod
    @property
    def argwhere(self) -> Callable:
        lib = self.load_lib()
        return lib.argwhere

    @classmethod
    @property
    def logical_or(self) -> Callable:
        lib = self.load_lib()
        return lib.logical_or

    @classmethod
    @property
    def logical_and(self) -> Callable:
        lib = self.load_lib()
        return lib.logical_and

    @classmethod
    @property
    def ones(self) -> Callable:
        lib = self.load_lib()
        return lib.ones

    @classmethod
    @property
    def stack(self) -> Callable:
        lib = self.load_lib()
        return lib.stack

    @classmethod
    @property
    def floor(self) -> Callable:
        lib = self.load_lib()
        return lib.floor

    @classmethod
    @property
    def log2(self) -> Callable:
        lib = self.load_lib()
        return lib.log2

    @classmethod
    @property
    def tan(self) -> Callable:
        lib = self.load_lib()
        return lib.tan

    @classmethod
    @property
    def arccos(self) -> Callable:
        lib = self.load_lib()
        return lib.arccos

    @classmethod
    @property
    def count_nonzero(self) -> Callable:
        lib = self.load_lib()
        return lib.count_nonzero

    @classmethod
    @property
    def zeros(self) -> Callable:
        lib = self.load_lib()
        return lib.zeros

    @classmethod
    @property
    def clip(self) -> Callable:
        lib = self.load_lib()
        return lib.clip

    @classmethod
    @property
    def pi(self) -> Callable:
        lib = self.load_lib()
        return lib.pi

    @classmethod
    @property
    def arange(self) -> Callable:
        lib = self.load_lib()
        return lib.arange

    @classmethod
    @property
    def concatenate(self) -> Callable:
        lib = self.load_lib()
        return lib.concatenate

    @classmethod
    @property
    def sum(self) -> Callable:
        lib = self.load_lib()
        return lib.sum

    @classmethod
    @property
    def inf(self) -> Callable:
        lib = self.load_lib()
        return lib.inf

    @classmethod
    @property
    def finfo(self) -> Callable:
        lib = self.load_lib()
        return lib.finfo

    @classmethod
    @property
    def allclose(self) -> Callable:
        lib = self.load_lib()
        return lib.allclose

    @classmethod
    @property
    def isclose(self) -> Callable:
        lib = self.load_lib()
        return lib.isclose
