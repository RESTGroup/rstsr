"""rstsr_faer.api — the array-API-standard namespace graded by array-api-tests.

Thin Python layer over the pyo3 module ``rstsr_faer.rstsr_faer``: signature
shims, protocol objects, and marshalling only.  No numeric algorithms live
here — everything numeric happens in Rust (rstsr, faer device).

Consciously outside the standard's surface (validation instrument, not a
product): names absent from this module are rstsr gaps, recorded in the gap
register rather than papered over with Python fallbacks.
"""

import builtins

from . import rstsr_faer as _pkg
from .rstsr_faer import (
    NativeArray,
    Dtype,
    Device,
    Finfo,
    Iinfo,
    asarray_from_flat,
    astype as _astype,
    zeros as _zeros,
    ones as _ones,
    empty as _empty,
    full as _full,
    arange as _arange,
    add as _add,
    subtract as _subtract,
    multiply as _multiply,
    divide as _divide,
    negative as _negative,
    abs as _abs,
    equal as _equal,
    not_equal as _not_equal,
    less as _less,
    less_equal as _less_equal,
    greater as _greater,
    greater_equal as _greater_equal,
    all as _all,
    any as _any,
    isnan as _isnan,
    isfinite as _isfinite,
    isinf as _isinf,
    sum as _sum,
    prod as _prod,
    max as _max,
    min as _min,
    mean as _mean,
    var as _var,
    std as _std,
    cumulative_sum as _cumulative_sum,
    cumulative_prod as _cumulative_prod,
    acos as _acos,
    acosh as _acosh,
    asin as _asin,
    asinh as _asinh,
    atan as _atan,
    atanh as _atanh,
    cos as _cos,
    cosh as _cosh,
    exp as _exp,
    expm1 as _expm1,
    log as _log,
    log2 as _log2,
    log10 as _log10,
    reciprocal as _reciprocal,
    sin as _sin,
    sinh as _sinh,
    sqrt as _sqrt,
    tan as _tan,
    tanh as _tanh,
    ceil as _ceil,
    floor as _floor,
    trunc as _trunc,
    round as _round,
    positive as _positive,
    square as _square,
    sign as _sign,
    conj as _conj,
    signbit as _signbit,
    real as _real,
    imag as _imag,
    invert as _invert,
    maximum as _maximum,
    minimum as _minimum,
    floor_divide as _floor_divide,
    atan2 as _atan2,
    copysign as _copysign,
    hypot as _hypot,
    nextafter as _nextafter,
    logaddexp as _logaddexp,
    remainder as _remainder,
    pow as _pow,
    bitwise_and as _bitwise_and,
    bitwise_or as _bitwise_or,
    bitwise_xor as _bitwise_xor,
    bitwise_left_shift as _bitwise_left_shift,
    bitwise_right_shift as _bitwise_right_shift,
    logical_and as _logical_and,
    logical_or as _logical_or,
    logical_xor as _logical_xor,
    broadcast_to as _broadcast_to,
    reshape as _reshape,
    transpose as _transpose,
    finfo as _finfo,
    iinfo as _iinfo,
)

__array_api_version__ = "2025.12"

# Python builtin types, captured before the dtype singletons below shadow
# `bool` (and keep int/float/complex aliases explicit for symmetry)
_py_bool = builtins.bool
_py_int = builtins.int
_py_float = builtins.float
_py_complex = builtins.complex

# ----------------------------------------------------------------- constants --

e = _pkg.e
inf = _pkg.inf
nan = _pkg.nan
pi = _pkg.pi

# -------------------------------------------------------------------- dtypes --

bool = _pkg.bool
int8 = _pkg.int8
int16 = _pkg.int16
int32 = _pkg.int32
int64 = _pkg.int64
uint8 = _pkg.uint8
uint16 = _pkg.uint16
uint32 = _pkg.uint32
uint64 = _pkg.uint64
float32 = _pkg.float32
float64 = _pkg.float64
complex64 = _pkg.complex64
complex128 = _pkg.complex128

# -------------------------------------------------------------------- device --

_DEVICE = _pkg.device_cpu

# Spec sentinel for indexing (x[..., newaxis, :]); `None` in an index tuple.
newaxis = None


def _check_device(device, /):
    if device is None:
        return
    if not isinstance(device, Device):
        raise TypeError(f"device must be None or the Device('cpu') singleton, got {device!r}")
    if device is not _DEVICE:
        raise ValueError(f"unsupported device {device!r}")


# --------------------------------------------------------------- namespace ----


def __array_namespace__(*, api_version=None):
    if api_version is not None and api_version != __array_api_version__:
        raise ValueError(f"unsupported api_version {api_version!r}")
    import sys

    return sys.modules[__name__]


class _NamespaceInfo:
    """Result of ``xp.__array_namespace_info__()`` (standard 2025.12)."""

    def __init__(self, api_version):
        self.api_version = api_version

    def devices(self, /):
        return [_DEVICE]

    def default_dtypes(self, /, *, device=None):
        _check_device(device)
        return {
            "real": float64,
            "integral": int64,
            "complex": complex128,
        }

    def dtypes(self, /, *, device=None, kind=None):
        _check_device(device)
        groups = {
            "bool": {"bool": bool},
            "integral": {
                "int8": int8, "int16": int16, "int32": int32, "int64": int64,
                "uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64,
            },
            "real floating": {"float32": float32, "float64": float64},
            "complex floating": {"complex64": complex64, "complex128": complex128},
        }
        if kind is None:
            kinds = tuple(groups)
        elif isinstance(kind, str):
            kinds = (kind,)
        else:
            kinds = tuple(kind)
        for k in kinds:
            if k not in groups:
                raise ValueError(f"unrecognized kind {k!r}")
        return {k: dict(groups[k]) for k in kinds}

    def capabilities(self, /):
        return {
            "boolean indexing": True,
            "data-dependent shapes": False,
            "max dimensions": 8,
        }


def __array_namespace_info__(*, api_version=None):
    if api_version is None:
        api_version = __array_api_version__
    if api_version != __array_api_version__:
        raise ValueError(f"unsupported api_version {api_version!r}")
    return _NamespaceInfo(api_version)


# ------------------------------------------------------------------- helpers --

_KIND_ORDER = {"bool": 0, "integral": 1, "real floating": 2, "complex floating": 3}


def _kind(dtype, /):
    return _pkg.dtype_kind(dtype)


def _unimplemented(name, /):
    raise NotImplementedError(
        f"{name}: not provided by rstsr (rstsr-faer-py gap) — recorded in the "
        f"gap register, no Python-side workaround exists"
    )


def _handle(x, /):
    if isinstance(x, Array):
        return x._h
    raise TypeError(f"expected an rstsr_faer.api Array, got {type(x).__name__}")


def _wrap(h, /):
    return Array.__new__(Array, h)


def _scalar_operand(value, ref_dtype, /):
    """Weak-scalar marshalling: cast a Python scalar to the reference dtype.

    Cross-kind scalars (scalar kind higher than the array kind) are declined —
    rstsr has no cross-dtype promotion (register G-009) and the shim carries
    no promotion table of its own.
    """
    if isinstance(value, Array):
        return value
    if not isinstance(value, (_py_bool, _py_int, _py_float, _py_complex)):
        raise TypeError(
            f"expected an rstsr_faer.api Array or a Python scalar, got {type(value).__name__}"
        )
    kind = (
        "bool" if isinstance(value, _py_bool)
        else "integral" if isinstance(value, _py_int)
        else "real floating" if isinstance(value, _py_float)
        else "complex floating"
    )
    own_kind = _kind(ref_dtype)
    if _KIND_ORDER[kind] > _KIND_ORDER[own_kind]:
        raise TypeError(
            f"cross-kind scalar promotion ({kind} scalar vs {own_kind} array) is not "
            f"provided by rstsr (gap G-009)"
        )
    return asarray(value, dtype=ref_dtype)


def _operands(x1, x2, /):
    """Handle pair for a binary namespace function, with weak-scalar support."""
    if isinstance(x1, Array):
        return x1._h, _scalar_operand(x2, x1.dtype)._h
    if isinstance(x2, Array):
        return _scalar_operand(x1, x2.dtype)._h, x2._h
    raise TypeError("at least one operand must be an rstsr_faer.api Array")


# --------------------------------------------------------------------- Array --


class Array:
    """array-API Array protocol object wrapping an opaque native tensor handle.

    Constructed only through namespace functions (asarray, zeros, ...); the
    ``__new__(cls, handle)`` form is internal to this module.
    """

    __slots__ = ("_h",)

    def __new__(cls, handle, /):
        if not isinstance(handle, NativeArray):
            raise TypeError("Array.__new__ expects a native tensor handle")
        self = super().__new__(cls)
        self._h = handle
        return self

    # ---- properties --------------------------------------------------

    @property
    def dtype(self, /):
        return self._h.dtype()

    @property
    def device(self, /):
        return _DEVICE

    @property
    def ndim(self, /):
        return self._h.ndim()

    @property
    def shape(self, /):
        return self._h.shape()

    @property
    def size(self, /):
        return self._h.size()

    @property
    def T(self, /):
        if self.ndim < 2:
            return self
        return permute_dims(self, tuple(range(self.ndim - 1, -1, -1)))

    @property
    def mT(self, /):
        if self.ndim < 2:
            return self
        axes = tuple(range(self.ndim))
        axes = axes[:-2] + (axes[-1], axes[-2])
        return permute_dims(self, axes)

    # ---- protocol methods ---------------------------------------------

    def __array_namespace__(self, /, *, api_version=None):
        return __array_namespace__(api_version=api_version)

    def __dlpack_device__(self, /):
        return (1, 0)

    def __dlpack__(self, /, *, stream=None, max_version=None, dl_device=None, copy=False):
        if stream is not None:
            raise ValueError("Array.__dlpack__: stream must be None on CPU devices")
        if dl_device is not None:
            try:
                dd = tuple(dl_device)
            except TypeError:
                raise TypeError("dl_device must be None or a (device_type, device_id) tuple") from None
            if dd != (1, 0):
                raise BufferError(
                    f"Array.__dlpack__: unsupported dl_device {dd} (only CPU (1, 0))"
                )
        if max_version is not None:
            try:
                mv = tuple(max_version)
            except TypeError:
                raise TypeError("max_version must be None or a (major, minor) tuple") from None
            if len(mv) != 2:
                raise TypeError("max_version must be a (major, minor) tuple")
        # Exports always carry a copy (IS_COPIED); a versioned capsule is
        # produced regardless of max_version (registered divergence).
        return _pkg.dlpack_export(self._h)

    def __getitem__(self, index, /):
        if isinstance(index, tuple) and builtins.len(index) == 0:
            return self  # () is a no-op index at any dimensionality
        key = index if isinstance(index, tuple) else (index,)
        if builtins.any(isinstance(k, Array) for k in key):
            _unimplemented("boolean-mask / integer-array indexing (rstsr gaps G-038/G-039)")
        return _wrap(_pkg.getitem_basic(self._h, key))

    def __setitem__(self, key, value, /):
        key = key if isinstance(key, tuple) else (key,)
        if builtins.any(isinstance(k, Array) for k in key):
            _unimplemented("boolean-mask item assignment (rstsr gap G-038)")
        if isinstance(value, Array):
            _pkg.setitem_basic(self._h, key, value._h)
        else:
            _pkg.setitem_scalar(self._h, key, value)

    def tolist(self, /):
        return self._h.tolist()

    def item(self, /):
        return self._h.item()

    def to_device(self, device, /, *, stream=None):
        _check_device(device)
        if stream is not None:
            raise ValueError("stream is not supported on cpu devices")
        return self

    def astype(self, dtype, /, *, copy=True):
        return _wrap(_astype(self._h, dtype, copy))

    # ---- python scalar conversion --------------------------------------

    def __bool__(self, /):
        if self.size != 1:
            raise TypeError(
                "bool(value) is ambiguous for arrays with more than one element"
            )
        return _py_bool(self.item())

    def __int__(self, /):
        if self.size != 1:
            raise TypeError("int(array) is only defined for size-1 arrays")
        return _py_int(self.item())

    def __float__(self, /):
        if self.size != 1:
            raise TypeError("float(array) is only defined for size-1 arrays")
        return _py_float(self.item())

    def __complex__(self, /):
        if self.size != 1:
            raise TypeError("complex(array) is only defined for size-1 arrays")
        return _py_complex(self.item())

    def __index__(self, /):
        if _kind(self.dtype) != "integral":
            raise TypeError("__index__ is only defined for integral arrays")
        return _py_int(self)

    def __len__(self, /):
        if self.ndim == 0:
            raise TypeError("len() of a 0-d array")
        return self.shape[0]

    def __repr__(self, /):
        # capped: failure reports embed reprs; huge tolists bloat them
        flat = self._h.tolist()
        if isinstance(flat, list) and len(flat) > 8:
            shown = flat[:8]
            tail = ", ..."
        else:
            shown, tail = flat, ""
        return f"Array({shown!r}{tail}, dtype={self.dtype!r}, shape={self.shape})"

    # ---- binary operators (weak-scalar forwarding; same-kind only) ------

    @staticmethod
    def _operand(other, /, own_dtype):
        """Marshal ``other`` for a binary op against ``self``.

        Returns an Array cast to ``own_dtype`` for weak scalars whose kind
        does not exceed the array's kind (spec weak promotion, within-kind);
        returns ``other`` unchanged when it is already an Array; returns None
        when ``other`` is not an operand (Python falls back to reflected).
        """
        if isinstance(other, Array):
            return other
        if isinstance(other, (_py_bool, _py_int, _py_float, _py_complex)):
            scalar_kind = (
                "bool" if isinstance(other, _py_bool)
                else "integral" if isinstance(other, _py_int)
                else "real floating" if isinstance(other, _py_float)
                else "complex floating"
            )
            own_kind = _kind(own_dtype)
            if _KIND_ORDER[scalar_kind] > _KIND_ORDER[own_kind]:
                raise TypeError(
                    f"cross-kind scalar promotion ({scalar_kind} scalar vs "
                    f"{own_kind} array) is not provided by rstsr (gap)"
                )
            return asarray(other, dtype=own_dtype)
        return None

    def _binary(self, other, op, reflected=False, /):
        other = self._operand(other, self.dtype)
        if other is None:
            return NotImplemented
        # mixed-dtype array pairs are passed through: the native layer serves
        # the pairs rstsr can promote and raises the gap otherwise (G-009)
        return op(self, other) if not reflected else op(other, self)

    # arithmetic
    def __add__(self, other, /):
        return self._binary(other, add)

    def __radd__(self, other, /):
        return self._binary(other, add, reflected=True)

    def __sub__(self, other, /):
        return self._binary(other, subtract)

    def __rsub__(self, other, /):
        return self._binary(other, subtract, reflected=True)

    def __mul__(self, other, /):
        return self._binary(other, multiply)

    def __rmul__(self, other, /):
        return self._binary(other, multiply, reflected=True)

    def __truediv__(self, other, /):
        return self._binary(other, divide)

    def __rtruediv__(self, other, /):
        return self._binary(other, divide, reflected=True)

    # comparison
    def __eq__(self, other, /):
        r = self._binary(other, equal)
        return False if r is NotImplemented else r

    def __ne__(self, other, /):
        r = self._binary(other, not_equal)
        return True if r is NotImplemented else r

    def __lt__(self, other, /):
        return self._binary(other, less)

    def __le__(self, other, /):
        return self._binary(other, less_equal)

    def __gt__(self, other, /):
        return self._binary(other, greater)

    def __ge__(self, other, /):
        return self._binary(other, greater_equal)

    # ---- arithmetic / bitwise operator dunders (W2) ---------------------

    def __pow__(self, other, /):
        return self._binary(other, pow)

    def __rpow__(self, other, /):
        return self._binary(other, pow, reflected=True)

    def __ipow__(self, other, /):
        return self._binary(other, pow)

    def __floordiv__(self, other, /):
        return self._binary(other, floor_divide)

    def __rfloordiv__(self, other, /):
        return self._binary(other, floor_divide, reflected=True)

    def __ifloordiv__(self, other, /):
        return self._binary(other, floor_divide)

    def __mod__(self, other, /):
        return self._binary(other, remainder)

    def __rmod__(self, other, /):
        return self._binary(other, remainder, reflected=True)

    def __imod__(self, other, /):
        return self._binary(other, remainder)

    def __and__(self, other, /):
        return self._binary(other, bitwise_and)

    def __rand__(self, other, /):
        return self._binary(other, bitwise_and, reflected=True)

    def __iand__(self, other, /):
        return self._binary(other, bitwise_and)

    def __or__(self, other, /):
        return self._binary(other, bitwise_or)

    def __ror__(self, other, /):
        return self._binary(other, bitwise_or, reflected=True)

    def __ior__(self, other, /):
        return self._binary(other, bitwise_or)

    def __xor__(self, other, /):
        return self._binary(other, bitwise_xor)

    def __rxor__(self, other, /):
        return self._binary(other, bitwise_xor, reflected=True)

    def __ixor__(self, other, /):
        return self._binary(other, bitwise_xor)

    def __lshift__(self, other, /):
        return self._binary(other, bitwise_left_shift)

    def __rlshift__(self, other, /):
        return self._binary(other, bitwise_left_shift, reflected=True)

    def __ilshift__(self, other, /):
        return self._binary(other, bitwise_left_shift)

    def __rshift__(self, other, /):
        return self._binary(other, bitwise_right_shift)

    def __rrshift__(self, other, /):
        return self._binary(other, bitwise_right_shift, reflected=True)

    def __irshift__(self, other, /):
        return self._binary(other, bitwise_right_shift)

    # ---- unary dunders --------------------------------------------------

    def __invert__(self, /):
        return bitwise_invert(self)

    def __neg__(self, /):
        return negative(self)

    def __pos__(self, /):
        return positive(self)

    def __abs__(self, /):
        return abs(self)

    __hash__ = None  # arrays are unhashable, like the reference implementations

    # ---- indexing (S1: absent by design; gap register) ------------------


# ------------------------------------------------------------- creation funcs --


def asarray(obj, /, *, dtype=None, device=None, copy=None):
    _check_device(device)
    if isinstance(obj, Array):
        if dtype is not None and dtype is not obj.dtype:
            out = _wrap(_astype(obj._h, dtype, True))
        elif copy:
            out = _wrap(obj._h.copy())
        else:
            out = obj
        return out
    if isinstance(obj, (_py_bool, _py_int, _py_float, _py_complex)):
        flat, shape = [obj], ()
    elif isinstance(obj, (list, tuple)):
        flat, shape = _flatten(obj)
    else:
        raise TypeError(
            f"asarray accepts scalars and (nested) Python lists; got {type(obj).__name__}"
        )
    if dtype is not None and not isinstance(dtype, Dtype):
        raise TypeError(f"dtype must be an xp dtype object, got {dtype!r}")
    return _wrap(asarray_from_flat(flat, shape, dtype, _DEVICE))


def _flatten(obj, /):
    """Marshalling: nested lists/tuples -> (flat list, shape). Raises on ragged."""
    shape = []
    probe = obj
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        if len(probe) == 0:
            break
        probe = probe[0]
    flat = []

    def rec(x, depth):
        if depth == len(shape) and not isinstance(x, (list, tuple)):
            if isinstance(x, (_py_bool, _py_int, _py_float, _py_complex)):
                flat.append(x)
                return
            raise TypeError(f"asarray: unsupported leaf {type(x).__name__}")
        if not isinstance(x, (list, tuple)) or len(x) != shape[depth]:
            raise ValueError("asarray: ragged nested sequences are not supported")
        for item in x:
            rec(item, depth + 1)

    rec(obj, 0)
    return flat, tuple(shape)


def zeros(shape, /, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_zeros(_norm_shape(shape), _dtype_or_default(dtype, "real"), _DEVICE))


def ones(shape, /, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_ones(_norm_shape(shape), _dtype_or_default(dtype, "real"), _DEVICE))


def empty(shape, /, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_empty(_norm_shape(shape), _dtype_or_default(dtype, "real"), _DEVICE))


def full(shape, fill_value, /, *, dtype=None, device=None):
    _check_device(device)
    if dtype is None:
        dtype = _infer_default(fill_value)
    if not isinstance(dtype, Dtype):
        raise TypeError(f"dtype must be an xp dtype object, got {dtype!r}")
    return _wrap(_full(_norm_shape(shape), fill_value, dtype, _DEVICE))


def arange(start, /, stop=None, step=1, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_arange(start, stop, step, dtype, _DEVICE))


def from_dlpack(obj, /, *, device=None, copy=False):
    _check_device(device)
    get = getattr(obj, "__dlpack__", None)
    if get is None:
        raise TypeError(f"from_dlpack: object of type {type(obj).__name__} has no __dlpack__ method")
    try:
        capsule = get(max_version=(1, 0))
    except TypeError:
        capsule = get()
    # Import always gathers into a fresh owned tensor (copy-only; registered).
    return _wrap(_pkg.dlpack_import(capsule))


def _norm_shape(shape, /):
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _infer_default(value, /):
    return _pkg.default_dtype_for(value)


def _dtype_or_default(dtype, kind, /):
    if dtype is None:
        return {"real": float64, "integral": int64, "complex": complex128, "bool": bool}[kind]
    if not isinstance(dtype, Dtype):
        raise TypeError(f"dtype must be an xp dtype object, got {dtype!r}")
    return dtype


# ------------------------------------------------------------ elementwise -----


def add(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_add(a, b))


def subtract(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_subtract(a, b))


def multiply(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_multiply(a, b))


def divide(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_divide(a, b))


def negative(x, /):
    return _wrap(_negative(_handle(x)))


def abs(x, /):
    return _wrap(_abs(_handle(x)))


# ------------------------------------------------------------- comparisons ----


def equal(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_equal(a, b))


def not_equal(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_not_equal(a, b))


def less(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_less(a, b))


def less_equal(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_less_equal(a, b))


def greater(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_greater(a, b))


def greater_equal(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_greater_equal(a, b))


# ----------------------------------------------------------- logical / tests --


def all(x, /, *, axis=None, keepdims=False):
    return _wrap(_all(_handle(x), _norm_axes(axis), keepdims))


def any(x, /, *, axis=None, keepdims=False):
    return _wrap(_any(_handle(x), _norm_axes(axis), keepdims))


def isnan(x, /):
    return _wrap(_isnan(_handle(x)))


def isfinite(x, /):
    return _wrap(_isfinite(_handle(x)))


def isinf(x, /):
    return _wrap(_isinf(_handle(x)))




# ------------------------------------------------- elementwise (W2 surface) --


def acos(x, /):
    return _wrap(_acos(_handle(x)))

def acosh(x, /):
    return _wrap(_acosh(_handle(x)))

def asin(x, /):
    return _wrap(_asin(_handle(x)))

def asinh(x, /):
    return _wrap(_asinh(_handle(x)))

def atan(x, /):
    return _wrap(_atan(_handle(x)))

def atanh(x, /):
    return _wrap(_atanh(_handle(x)))

def ceil(x, /):
    return _wrap(_ceil(_handle(x)))

def conj(x, /):
    return _wrap(_conj(_handle(x)))

def cos(x, /):
    return _wrap(_cos(_handle(x)))

def cosh(x, /):
    return _wrap(_cosh(_handle(x)))

def exp(x, /):
    return _wrap(_exp(_handle(x)))

def expm1(x, /):
    return _wrap(_expm1(_handle(x)))

def floor(x, /):
    return _wrap(_floor(_handle(x)))

def imag(x, /):
    return _wrap(_imag(_handle(x)))

def log(x, /):
    return _wrap(_log(_handle(x)))

def log2(x, /):
    return _wrap(_log2(_handle(x)))

def log10(x, /):
    return _wrap(_log10(_handle(x)))

def real(x, /):
    return _wrap(_real(_handle(x)))

def reciprocal(x, /):
    return _wrap(_reciprocal(_handle(x)))

def round(x, /):
    return _wrap(_round(_handle(x)))

def positive(x, /):
    return _wrap(_positive(_handle(x)))


def sign(x, /):
    return _wrap(_sign(_handle(x)))

def signbit(x, /):
    return _wrap(_signbit(_handle(x)))

def sin(x, /):
    return _wrap(_sin(_handle(x)))

def sinh(x, /):
    return _wrap(_sinh(_handle(x)))

def square(x, /):
    return _wrap(_square(_handle(x)))


def sqrt(x, /):
    return _wrap(_sqrt(_handle(x)))

def tan(x, /):
    return _wrap(_tan(_handle(x)))

def tanh(x, /):
    return _wrap(_tanh(_handle(x)))

def trunc(x, /):
    return _wrap(_trunc(_handle(x)))


# ------------------------------------------------- inverse / bitwise helpers --


def logical_not(x, /):
    h = _handle(x)
    if _kind(h.dtype()) != "bool":
        raise TypeError("logical_not: only boolean arrays are allowed")
    return _wrap(_invert(h))


def bitwise_invert(x, /):
    return _wrap(_invert(_handle(x)))


def maximum(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_maximum(a, b))

def minimum(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_minimum(a, b))

def floor_divide(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_floor_divide(a, b))

def atan2(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_atan2(a, b))

def copysign(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_copysign(a, b))

def hypot(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_hypot(a, b))

def nextafter(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_nextafter(a, b))

def logaddexp(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_logaddexp(a, b))

def remainder(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_remainder(a, b))

def pow(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_pow(a, b))

def bitwise_and(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_bitwise_and(a, b))

def bitwise_or(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_bitwise_or(a, b))

def bitwise_xor(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_bitwise_xor(a, b))

def bitwise_left_shift(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_bitwise_left_shift(a, b))

def bitwise_right_shift(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_bitwise_right_shift(a, b))

def logical_and(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_logical_and(a, b))

def logical_or(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_logical_or(a, b))

def logical_xor(x1, x2, /):
    a, b = _operands(x1, x2)
    return _wrap(_logical_xor(a, b))


# ------------------------------------------------------- statistical (W3) -----
#
# Axes reductions over the rstsr `*_with_args` families. The accumulation
# dtype rule (standard 2025.12: integer inputs widen to the default integer
# dtype unless dtype= is given; floats/complexes keep their dtype) is applied
# by casting the input with the existing astype path BEFORE the same-dtype
# reduction — the order the standard itself recommends.


def _norm_axes(axis, /):
    """Marshalling: axis kwarg (None | int | tuple of ints) -> list | None."""
    if axis is None:
        return None
    if isinstance(axis, _py_int):
        return [axis]
    if isinstance(axis, tuple) and builtins.all(isinstance(a, _py_int) for a in axis):
        return list(axis)
    raise TypeError(f"axis must be None, an int, or a tuple of ints, got {axis!r}")


def _accumulation_dtype(x, dtype, /):
    """Resolve the accumulation result dtype (spec sum/prod/cumulative rule)."""
    if dtype is not None:
        if not isinstance(dtype, Dtype):
            raise TypeError(f"dtype must be an xp dtype object, got {dtype!r}")
        return dtype
    kind = _kind(x.dtype)
    if kind in ("bool", "integral"):
        return uint64 if x.dtype.name in ("uint8", "uint16", "uint32", "uint64") else int64
    return x.dtype


def _accumulated(x, dtype, /):
    """Cast x to the accumulation dtype when it differs (no-op otherwise)."""
    out = _accumulation_dtype(x, dtype)
    return x if out is x.dtype else astype(x, out)


def sum(x, /, *, axis=None, dtype=None, keepdims=False):
    y = _accumulated(x, dtype)
    return _wrap(_sum(_handle(y), _norm_axes(axis), keepdims))


def prod(x, /, *, axis=None, dtype=None, keepdims=False):
    y = _accumulated(x, dtype)
    return _wrap(_prod(_handle(y), _norm_axes(axis), keepdims))


def max(x, /, *, axis=None, keepdims=False):
    return _wrap(_max(_handle(x), _norm_axes(axis), keepdims))


def min(x, /, *, axis=None, keepdims=False):
    return _wrap(_min(_handle(x), _norm_axes(axis), keepdims))


def _float_input(x, /):
    """mean/std/var input marshalling: integers (and bool) promote to the
    default real floating dtype (spec), floats/complexes pass through."""
    if _kind(_handle(x).dtype()) in ("bool", "integral"):
        return astype(x, float64)
    return x


def mean(x, /, *, axis=None, keepdims=False):
    return _wrap(_mean(_handle(_float_input(x)), _norm_axes(axis), keepdims))


def var(x, /, *, axis=None, correction=0.0, keepdims=False):
    y = _float_input(x)
    return _wrap(_var(_handle(y), _norm_axes(axis), float(correction), keepdims))


def std(x, /, *, axis=None, correction=0.0, keepdims=False):
    y = _float_input(x)
    return _wrap(_std(_handle(y), _norm_axes(axis), float(correction), keepdims))


def cumulative_sum(x, /, *, axis=None, dtype=None, include_initial=False):
    y = _accumulated(x, dtype)
    return _wrap(_cumulative_sum(_handle(y), axis, include_initial))


def cumulative_prod(x, /, *, axis=None, dtype=None, include_initial=False):
    y = _accumulated(x, dtype)
    return _wrap(_cumulative_prod(_handle(y), axis, include_initial))


def broadcast_to(x, /, shape):
    return _wrap(_broadcast_to(_handle(x), _norm_shape(shape)))



# ------------------------------------------------------------- manipulation ---


def reshape(x, /, shape, *, copy=None):
    return _wrap(_reshape(_handle(x), _norm_shape(shape)))


def permute_dims(x, /, axes):
    return _wrap(_transpose(_handle(x), tuple(axes)))


# --------------------------------------------------------------- data types ---


def astype(x, /, dtype, *, copy=True):
    return _wrap(_astype(_handle(x), dtype, copy))


def finfo(type, /):
    if isinstance(type, Array):
        type = type.dtype
    return _finfo(type)


def iinfo(type, /):
    if isinstance(type, Array):
        type = type.dtype
    return _iinfo(type)


# -------------------------------------------------------------------- export --

__all__ = [
    "__array_api_version__",
    "__array_namespace__",
    "__array_namespace_info__",
    # constants
    "e", "inf", "nan", "pi",
    # dtypes
    "bool", "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "float32", "float64", "complex64", "complex128",
    # creation
    "asarray", "zeros", "ones", "empty", "full", "arange", "from_dlpack",
    # elementwise
    "add", "subtract", "multiply", "divide", "negative", "abs", "positive",
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil",
    "conj", "copysign", "cos", "cosh", "exp", "expm1", "floor", "floor_divide",
    "hypot", "imag", "log", "log2", "log10", "logaddexp", "maximum",
    "minimum", "nextafter", "pow", "real", "reciprocal", "remainder", "round",
    "sign", "signbit", "sin", "sinh", "sqrt", "square", "tan", "tanh", "trunc",
    "bitwise_and", "bitwise_left_shift", "bitwise_invert", "bitwise_or",
    "bitwise_right_shift", "bitwise_xor", "logical_and", "logical_not",
    "logical_or", "logical_xor",
    # comparison
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    # logical / tests
    "all", "any", "isnan", "isfinite", "isinf",
    # statistical (axes reductions + cumulative scans)
    "sum", "prod", "max", "min", "mean", "std", "var",
    "cumulative_sum", "cumulative_prod",
    # broadcasting
    "broadcast_to",
    # manipulation
    "reshape", "permute_dims",
    # data types
    "astype", "finfo", "iinfo",
    # constants / sentinels
    "newaxis",
]
