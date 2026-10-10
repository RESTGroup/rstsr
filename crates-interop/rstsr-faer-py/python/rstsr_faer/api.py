"""rstsr_faer.api — the array-API-standard namespace graded by array-api-tests.

Thin Python layer over the pyo3 module ``rstsr_faer.rstsr_faer``: signature
shims, protocol objects, and marshalling only.  No numeric algorithms live
here — everything numeric happens in Rust (rstsr, faer device).

One corner where the standard asks for more than Rust does is *dtype
promotion queries* (``result_type``, ``can_cast``, ``isdtype``). In Rust,
promotion is a compile-time relation carried by a trait bound
(``DTypePromoteAPI``), so there is nothing to call at runtime; a Python
namespace has only dtype tokens, so the standard mandates explicit functions.
They resolve through ``rstsr_faer.rstsr_faer.dtype_promote``, which
re-projects that same trait lattice onto name pairs — see ``src/promotion.rs``.

Consciously outside the standard's surface (validation instrument, not a
product): names absent from this module are rstsr gaps, recorded in the gap
register rather than papered over with Python fallbacks.
"""

import builtins
import collections

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
    clip as _clip,
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
    log1p as _log1p,
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
    eye as _eye,
    linspace as _linspace,
    tril as _tril,
    triu as _triu,
    meshgrid as _meshgrid,
    broadcast_shapes as _broadcast_shapes,
    concat as _concat,
    stack as _stack,
    unstack as _unstack,
    expand_dims as _expand_dims,
    squeeze as _squeeze,
    flip as _flip,
    moveaxis as _moveaxis,
    argmax as _argmax,
    argmin as _argmin,
    count_nonzero as _count_nonzero,
    sum_bool as _sum_bool,
    where as _where,
    take as _take,
    repeat as _repeat,
    roll as _roll,
    tile as _tile,
    diff as _diff,
    sort as _sort,
    argsort as _argsort,
    searchsorted as _searchsorted,
    nonzero as _nonzero,
    isin as _isin,
    unique_values as _unique_values,
    unique_counts as _unique_counts,
    unique_inverse as _unique_inverse,
    unique_all as _unique_all,
    take_along_axis as _take_along_axis,
    linalg_cholesky as _linalg_cholesky,
    linalg_det as _linalg_det,
    linalg_diagonal as _linalg_diagonal,
    linalg_eigh as _linalg_eigh,
    linalg_eigvalsh as _linalg_eigvalsh,
    linalg_inv as _linalg_inv,
    linalg_matmul as _linalg_matmul,
    linalg_matrix_transpose as _linalg_matrix_transpose,
    linalg_pinv as _linalg_pinv,
    linalg_solve as _linalg_solve,
    linalg_svd as _linalg_svd,
    linalg_svdvals as _linalg_svdvals,
    linalg_slogdet as _linalg_slogdet,
    linalg_vecdot as _linalg_vecdot,
    linalg_tensordot as _linalg_tensordot,
)

__array_api_version__ = "2025.12"

# Python builtin types, captured before the dtype singletons below shadow
# `bool` (and keep int/float/complex aliases explicit for symmetry)
_py_bool = builtins.bool
_py_int = builtins.int
_py_float = builtins.float
_py_complex = builtins.complex
_py_max = builtins.max

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
        return (_DEVICE,)

    def default_device(self, /):
        return _DEVICE

    def default_dtypes(self, /, *, device=None):
        _check_device(device)
        return {
            "real floating": float64,
            "integral": int64,
            "complex floating": complex128,
            # 2025.12 key: every index output is lifted to int64 (idx_lift)
            "indexing": int64,
        }

    def dtypes(self, /, *, device=None, kind=None):
        _check_device(device)
        groups = {
            "bool": {"bool": bool},
            "signed integer": {
                "int8": int8, "int16": int16, "int32": int32, "int64": int64,
            },
            "unsigned integer": {
                "uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64,
            },
            "real floating": {"float32": float32, "float64": float64},
            "complex floating": {"complex64": complex64, "complex128": complex128},
        }
        # composite kinds: 'integral' and 'numeric' are unions of the atomic ones
        groups["integral"] = {**groups["signed integer"], **groups["unsigned integer"]}
        groups["numeric"] = {
            **groups["integral"], **groups["real floating"], **groups["complex floating"],
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
        return {name: dt for k in kinds for name, dt in groups[k].items()}

    def capabilities(self, /):
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
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


def _is_single_bool_mask(key, /):
    """True when `key` is a lone boolean Array (whole-tensor mask indexing)."""
    return builtins.len(key) == 1 and isinstance(key[0], Array) and _kind(key[0].dtype) == "bool"


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
        # A 0-d integer array is a scalar index (NumPy semantics), not
        # integer-array indexing: marshal it through __index__.
        key = tuple(
            _py_int(k)
            if isinstance(k, Array) and k.ndim == 0 and _kind(k.dtype) == "integral"
            else k
            for k in key
        )
        if builtins.any(isinstance(k, Array) for k in key):
            # A lone boolean array is whole-tensor mask indexing; any other
            # array key rides rust-side array indexing (`rt::array_index`),
            # which accepts mixed basic + integer-array indexers.
            if _is_single_bool_mask(key):
                return _wrap(_pkg.getitem_mask(self._h, key[0]._h))
            key = tuple(k._h if isinstance(k, Array) else k for k in key)
        return _wrap(_pkg.getitem_basic(self._h, key))

    def __setitem__(self, key, value, /):
        key = key if isinstance(key, tuple) else (key,)
        # a 0-d integer array is a scalar index, as in __getitem__
        key = tuple(
            _py_int(k)
            if isinstance(k, Array) and k.ndim == 0 and _kind(k.dtype) == "integral"
            else k
            for k in key
        )
        if builtins.any(isinstance(k, Array) for k in key):
            if _is_single_bool_mask(key):
                # A lone boolean mask rides mask_fill for a scalar value or a
                # size-1 array value. Any larger array value rides array-index
                # assignment, which lowers a mask only when it has at least one
                # dimension -- a 0-d mask is rejected there (not lowerable).
                if isinstance(value, Array) and value.size > 1:
                    _pkg.setitem_array(self._h, (key[0]._h,), value._h)
                elif isinstance(value, Array):
                    _pkg.setitem_mask(self._h, key[0]._h, value._h)
                else:
                    _pkg.setitem_mask_scalar(self._h, key[0]._h, value)
                return
            # integer-array keys (masks mixed with basic indexers included) ride
            # rust-side array-indexing assignment (`rt::array_index_assign`)
            key = tuple(k._h if isinstance(k, Array) else k for k in key)
            if isinstance(value, Array):
                _pkg.setitem_array(self._h, key, value._h)
            else:
                _pkg.setitem_array_scalar(self._h, key, value)
            return
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

    def astype(self, dtype, /, *, copy=True, device=None):
        _check_device(device)
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

    # matrix product (PEP 465); same semantics as the top-level matmul
    def __matmul__(self, other, /):
        return matmul(self, other)

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


def zeros(shape, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_zeros(_norm_shape(shape), _dtype_or_default(dtype, "real"), _DEVICE))


def ones(shape, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_ones(_norm_shape(shape), _dtype_or_default(dtype, "real"), _DEVICE))


def empty(shape, *, dtype=None, device=None):
    _check_device(device)
    return _wrap(_empty(_norm_shape(shape), _dtype_or_default(dtype, "real"), _DEVICE))


def full(shape, fill_value, *, dtype=None, device=None):
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


# ---- creation: `*_like` family (marshalling only, same creation paths) -----


def _like_target(x, dtype, /):
    """Resolve (shape, dtype) for the `*_like` family from an input array."""
    if not isinstance(x, Array):
        raise TypeError(f"expected an rstsr_faer.api Array, got {type(x).__name__}")
    if dtype is None:
        return x.shape, x.dtype
    if not isinstance(dtype, Dtype):
        raise TypeError(f"dtype must be an xp dtype object, got {dtype!r}")
    return x.shape, dtype


def empty_like(x, /, *, dtype=None, device=None):
    _check_device(device)
    shape, dtype = _like_target(x, dtype)
    return _wrap(_empty(shape, dtype, _DEVICE))


def zeros_like(x, /, *, dtype=None, device=None):
    _check_device(device)
    shape, dtype = _like_target(x, dtype)
    return _wrap(_zeros(shape, dtype, _DEVICE))


def ones_like(x, /, *, dtype=None, device=None):
    _check_device(device)
    shape, dtype = _like_target(x, dtype)
    return _wrap(_ones(shape, dtype, _DEVICE))


def full_like(x, /, fill_value, *, dtype=None, device=None):
    _check_device(device)
    shape, dtype = _like_target(x, dtype)
    return _wrap(_full(shape, fill_value, dtype, _DEVICE))


def eye(n_rows, n_cols=None, /, *, k=0, dtype=None, device=None):
    _check_device(device)
    return _wrap(_eye(n_rows, n_cols, k, _dtype_or_default(dtype, "real")))


def linspace(start, stop, /, num, *, dtype=None, device=None, endpoint=True):
    _check_device(device)
    if dtype is not None and not isinstance(dtype, Dtype):
        raise TypeError(f"dtype must be an xp dtype object, got {dtype!r}")
    return _wrap(_linspace(start, stop, num, _py_bool(endpoint), dtype))


def tril(x, /, *, k=0):
    return _wrap(_tril(_handle(x), k))


def triu(x, /, *, k=0):
    return _wrap(_triu(_handle(x), k))


def meshgrid(*arrays, indexing="xy"):
    if indexing not in ("xy", "ij"):
        raise ValueError(f"meshgrid: indexing must be 'xy' or 'ij', got {indexing!r}")
    return tuple(_wrap(h) for h in _meshgrid([_handle(a) for a in arrays], indexing))


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
    h = _handle(x)
    kind = _kind(h.dtype())
    if kind in ("real floating", "complex floating"):
        return _wrap(_isnan(h))
    if kind in ("integral", "bool"):
        return zeros(h.shape(), dtype=bool)  # int/bool are never NaN
    raise TypeError(f"isnan: only numeric dtypes are allowed, got {kind}")


def isfinite(x, /):
    h = _handle(x)
    kind = _kind(h.dtype())
    if kind in ("real floating", "complex floating"):
        return _wrap(_isfinite(h))
    if kind in ("integral", "bool"):
        return ones(h.shape(), dtype=bool)  # int/bool are always finite
    raise TypeError(f"isfinite: only numeric dtypes are allowed, got {kind}")


def isinf(x, /):
    h = _handle(x)
    kind = _kind(h.dtype())
    if kind in ("real floating", "complex floating"):
        return _wrap(_isinf(h))
    if kind in ("integral", "bool"):
        return zeros(h.shape(), dtype=bool)  # int/bool are never inf
    raise TypeError(f"isinf: only numeric dtypes are allowed, got {kind}")




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

def log1p(x, /):
    return _wrap(_log1p(_handle(x)))

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


# ---- manipulation (W4): joins, splits, axis moves, broadcasting -----------


def _axes_vec(axis, opname, /):
    """Required axis argument (int or tuple of ints) -> list of ints."""
    axes = _norm_axes(axis)
    if axes is None:
        raise TypeError(f"{opname}: axis must be an int or a tuple of ints")
    return axes


def _norm_shape_ints(shape, opname, /):
    try:
        dims = list(shape)
    except TypeError:
        raise TypeError(f"{opname}: each shape must be a tuple/list of ints, got {shape!r}") from None
    for d in dims:
        if not isinstance(d, _py_int) or isinstance(d, _py_bool):
            raise TypeError(f"{opname}: shape dimensions must be ints, got {d!r}")
    return dims


def broadcast_shapes(*shapes):
    return tuple(_broadcast_shapes([_norm_shape_ints(s, "broadcast_shapes") for s in shapes]))


def broadcast_arrays(*arrays):
    """Broadcast arrays against each other.

    Spec composition of two rstsr primitives: the common shape comes from
    rstsr's own broadcasting rule (`broadcast_shapes`), each result from
    `broadcast_to` — so unlike rstsr's same-dtype `broadcast_arrays`, every
    input keeps its own dtype here.
    """
    if builtins.len(arrays) == 0:
        raise ValueError("broadcast_arrays: at least one array is required")
    hs = [_handle(a) for a in arrays]
    shape = tuple(_broadcast_shapes([h.shape() for h in hs]))
    return tuple(_wrap(_broadcast_to(h, shape)) for h in hs)


def _join_parts(arrays, opname, /):
    if not isinstance(arrays, (list, tuple)):
        raise TypeError(f"{opname}: arrays must be a list or tuple of arrays")
    if builtins.len(arrays) == 0:
        raise ValueError(f"{opname}: at least one array is required")
    return [_handle(a) for a in arrays]


def concat(arrays, /, *, axis=0):
    parts = _join_parts(arrays, "concat")
    # cross-dtype joins cast each part to the promoted common dtype first
    parts = _cast_parts(parts, "concat")
    if axis is None:
        # spec: axis=None flattens every array before concatenation
        parts = [_reshape(h, (-1,)) for h in parts]
        axis = 0
    return _wrap(_concat(parts, axis))


def stack(arrays, /, *, axis=0):
    parts = _cast_parts(_join_parts(arrays, "stack"), "stack")
    return _wrap(_stack(parts, axis))


def unstack(x, /, *, axis=0):
    return tuple(_wrap(h) for h in _unstack(_handle(x), axis))


def expand_dims(x, /, axis):
    return _wrap(_expand_dims(_handle(x), _axes_vec(axis, "expand_dims")))


def squeeze(x, /, axis=None):
    return _wrap(_squeeze(_handle(x), _norm_axes(axis)))


def flip(x, /, *, axis=None):
    return _wrap(_flip(_handle(x), _norm_axes(axis)))


def moveaxis(x, source, destination, /):
    return _wrap(
        _moveaxis(
            _handle(x),
            _axes_vec(source, "moveaxis"),
            _axes_vec(destination, "moveaxis"),
        )
    )


# ------------------------------------------- searching / indexing (W5) ------


def _axis_or_none(axis, opname, /):
    """argmax/argmin's axis: None or a single int (the standard has no tuple form)."""
    if axis is None:
        return None
    if isinstance(axis, _py_int) and not isinstance(axis, _py_bool):
        return axis
    raise TypeError(f"{opname}: axis must be None or an int, got {axis!r}")


def argmax(x, /, *, axis=None, keepdims=False):
    return _wrap(_argmax(_handle(x), _axis_or_none(axis, "argmax"), _py_bool(keepdims)))


def argmin(x, /, *, axis=None, keepdims=False):
    return _wrap(_argmin(_handle(x), _axis_or_none(axis, "argmin"), _py_bool(keepdims)))


def count_nonzero(x, /, *, axis=None, keepdims=False):
    """Count non-zero elements.

    Documented exception: for boolean input, rstsr's generic `count_nonzero`
    kernel is unavailable (`Zero` has no bool impl). Bool is instead routed to
    rstsr's **bool-specialized sum** (`TensorSumBoolAPI::sum_with_args_f`,
    `TOut = usize`; see `ops::sum_bool`) — counting `True`s is the 0/1 sum by
    definition. This is a rust-backed kernel, not a Python-side fallback.
    """
    h = _handle(x)
    axes = _norm_axes(axis)
    if _kind(h.dtype()) == "bool":
        return _wrap(_sum_bool(h, axes, _py_bool(keepdims)))
    return _wrap(_count_nonzero(h, axes, _py_bool(keepdims)))


def where(condition, x1, x2, /):
    """Element-wise select: x1 where ``condition`` is True, x2 elsewhere.

    The condition must be an Array with boolean dtype (the spec's "should have
    a boolean data type"; non-bool conditions are declined, never truthiness-
    cast). ``x1``/``x2`` may each be an Array or a Python scalar; scalars are
    converted to 0-d arrays of the array operand's dtype before the native call
    (spec "Mixing arrays with Python scalars" — the same weak-scalar rule as
    the binary functions, via ``_operands``). At least one of ``x1``/``x2``
    must be an Array; dtype promotion of Array pairs happens rust-side in
    rstsr's own matrix.
    """
    if not isinstance(condition, Array):
        raise TypeError(
            f"where: condition must be an rstsr_faer.api Array, got {type(condition).__name__}"
        )
    if _kind(condition.dtype) != "bool":
        raise TypeError(
            f"where: condition must have a boolean data type, got {condition.dtype!r}"
        )
    a, b = _operands(x1, x2)
    return _wrap(_where(condition._h, a, b))


def _clip_bound(value, ref, /):
    """Bound marshalling for ``clip``.

    The array-API ``clip`` keeps the input dtype (the bounds do not promote), so
    an Array bound is cast to ``x``'s dtype and a Python scalar is routed through
    ``_scalar_operand`` — the same weak-scalar rule (and cross-kind guard) as the
    other elementwise functions.
    """
    dtype = ref.dtype()
    if isinstance(value, Array):
        return value.astype(dtype, copy=False)._h
    return _scalar_operand(value, dtype)._h


def clip(x, /, min=None, max=None):
    """Clamp each element of ``x`` to the range ``[min, max]``.

    Either bound may be ``None`` (no bound on that side); with both absent the
    result is a copy of ``x`` (per the spec). A bound may be an Array
    (broadcast against ``x``) or a Python scalar. Per the spec, ``min`` and
    ``max`` do not participate in dtype promotion: both are cast to ``x``'s
    dtype, so the result keeps ``x``'s dtype.
    """
    h = _handle(x)
    lo = None if min is None else _clip_bound(min, h)
    hi = None if max is None else _clip_bound(max, h)
    return _wrap(_clip(h, lo, hi))


def take(x, /, indices, *, axis=None):
    h = _handle(x)
    if not isinstance(indices, Array):
        raise TypeError(f"take: indices must be an rstsr_faer.api Array, got {type(indices).__name__}")
    if _kind(indices.dtype) != "integral":
        raise TypeError(f"take: indices must have an integer data type, got {indices.dtype!r}")
    if indices.ndim != 1:
        raise ValueError(f"take: indices must be one-dimensional, got ndim={indices.ndim}")
    if axis is None:
        if h.ndim() != 1:
            raise ValueError("take: axis is required when x has more than one axis")
        axis = 0
    return _wrap(_take(h, indices.tolist(), axis))


def take_along_axis(x, /, indices, *, axis=-1):
    h = _handle(x)
    if not isinstance(indices, Array):
        raise TypeError(
            f"take_along_axis: indices must be an rstsr_faer.api Array, got {type(indices).__name__}"
        )
    if _kind(indices.dtype) != "integral":
        raise TypeError(f"take_along_axis: indices must have an integer data type, got {indices.dtype!r}")
    # shape compatibility (same ndim; broadcast-compatible outside `axis`) is
    # validated rust-side
    # tolist flattens nested ints; the shape travels alongside so the shim can
    # rebuild the index tensor (the `take` marshalling precedent)
    flat = indices.tolist()
    return _wrap(_take_along_axis(h, _flatten_indices(flat, indices.shape), list(indices.shape), axis))


def _flatten_indices(obj, shape, /):
    """Nested int lists -> flat int list (row-major), len == prod(shape)."""
    flat = []

    def rec(x, depth):
        if depth == builtins.len(shape):
            if isinstance(x, _py_int) and not isinstance(x, _py_bool):
                flat.append(x)
                return
            raise TypeError(f"indices leaves must be ints, got {type(x).__name__}")
        if not isinstance(x, (list, tuple)) or builtins.len(x) != shape[depth]:
            raise ValueError("indices: ragged nested sequences are not supported")
        for item in x:
            rec(item, depth + 1)

    rec(obj, 0)
    return flat


# ------------------------------------------ manip/sort/set wave (W6-8) -------


def _int_arg(value, opname, /):
    """Required int argument (bool rejected)."""
    if isinstance(value, _py_int) and not isinstance(value, _py_bool):
        return value
    raise TypeError(f"{opname}: expected an int, got {value!r}")


def _int_tuple_arg(value, opname, /):
    """int or tuple of ints -> list of ints."""
    if isinstance(value, _py_int) and not isinstance(value, _py_bool):
        return [value]
    if isinstance(value, tuple) and builtins.all(
        isinstance(s, _py_int) and not isinstance(s, _py_bool) for s in value
    ):
        return list(value)
    raise TypeError(f"{opname}: expected an int or a tuple of ints, got {value!r}")


def repeat(x, /, repeats, *, axis=None):
    h = _handle(x)
    if isinstance(repeats, Array):
        # dtype/ndim checks stand in for the rust boundary: repeats travel as a
        # flat unsigned int list, so bad dtypes/shapes get a clean error first
        if _kind(repeats.dtype) != "integral":
            raise TypeError(f"repeat: repeats must have an integer data type, got {repeats.dtype!r}")
        if repeats.ndim != 1:
            raise ValueError(f"repeat: repeats must be one-dimensional, got ndim={repeats.ndim}")
        reps = repeats.tolist()
        # negative repetitions are a ValueError (NumPy parity); a raw negative
        # would die as OverflowError at the unsigned pyo3 boundary
        if builtins.any(s < 0 for s in reps):
            raise ValueError(f"repeat: negative repeats are not allowed, got {reps!r}")
    elif isinstance(repeats, _py_int) and not isinstance(repeats, _py_bool):
        # bool is rejected (NumPy parity); a raw negative would die as
        # OverflowError at the unsigned pyo3 boundary
        if repeats < 0:
            raise ValueError(f"repeat: negative repeats are not allowed, got {repeats!r}")
        reps = None
    else:
        raise TypeError(
            f"repeat: repeats must be an int or an rstsr_faer.api Array, got {type(repeats).__name__}"
        )
    if axis is not None:
        axis = _int_arg(axis, "repeat(axis)")
    return _wrap(_repeat(h, reps, repeats if isinstance(repeats, _py_int) else None, axis))


def roll(x, /, shift=None, *, axis=None):
    h = _handle(x)
    shifts = _int_tuple_arg(shift, "roll(shift)")
    axes = None
    if axis is not None:
        axes = _int_tuple_arg(axis, "roll(axis)")
    # shift/axis lengths follow NumPy's broadcast rule; incompatibilities are
    # validated rust-side (surfaced as ValueError)
    return _wrap(_roll(h, shifts, axes))


def tile(x, /, repetitions):
    h = _handle(x)
    reps = _int_tuple_arg(repetitions, "tile(repetitions)")
    # negative sizes are a ValueError (NumPy parity); a raw negative would die
    # as OverflowError in the unsigned pyo3 boundary
    if builtins.any(s < 0 for s in reps):
        raise ValueError(f"tile: negative repetitions are not allowed, got {reps!r}")
    return _wrap(_tile(h, reps))


def diff(x, /, *, axis=-1, n=1, prepend=None, append=None):
    h = _handle(x)
    axis = _int_arg(axis, "diff(axis)")
    # n = 0 returns the input unchanged (NumPy parity; rust diff_f handles it)
    if not isinstance(n, _py_int) or isinstance(n, _py_bool) or n < 0:
        raise ValueError(f"diff: n must be a non-negative integer, got {n!r}")
    pre = _handle(prepend) if prepend is not None else None
    app = _handle(append) if append is not None else None
    return _wrap(_diff(h, axis, n, pre, app))


def sort(x, /, *, axis=-1, descending=False, stable=True):
    return _wrap(_sort(_handle(x), _int_arg(axis, "sort(axis)"), _py_bool(descending), _py_bool(stable)))


def argsort(x, /, *, axis=-1, descending=False, stable=True):
    return _wrap(
        _argsort(_handle(x), _int_arg(axis, "argsort(axis)"), _py_bool(descending), _py_bool(stable))
    )


def searchsorted(x1, /, x2, *, side="left", sorter=None):
    h1 = _handle(x1)
    if side not in ("left", "right"):
        raise ValueError(f"searchsorted: side must be 'left' or 'right', got {side!r}")
    if isinstance(x2, Array):
        h2 = x2._h
    elif isinstance(x2, (_py_int, _py_float)) and not isinstance(x2, _py_bool):
        # 2025.12 mixing-scalars rule: a compatible scalar becomes a 0-d array
        # of x1's dtype. Compatibility: int scalar for an int array; int/float
        # scalar for a real-floating array. A float scalar with an int array is
        # unspecified by the standard (may promote or raise) — rstsr's
        # searchsorted is single-dtype, so it raises (same as a float *array*).
        if _kind(h1.dtype()) == "integral" and isinstance(x2, _py_float):
            raise TypeError(
                f"searchsorted: float scalar x2 with an integer array x1 is not "
                f"provided by rstsr (gap G-009); cast x2 or x1 first"
            )
        h2 = asarray(x2, dtype=h1.dtype())._h
    else:
        raise TypeError(
            f"searchsorted: x2 must be an rstsr_faer.api Array, int, or float, "
            f"got {type(x2).__name__}"
        )
    sort_idx = None
    if sorter is not None:
        if not isinstance(sorter, Array):
            raise TypeError(
                f"searchsorted: sorter must be an rstsr_faer.api Array, got {type(sorter).__name__}"
            )
        if _kind(sorter.dtype) != "integral":
            raise TypeError(f"searchsorted: sorter must have an integer data type, got {sorter.dtype!r}")
        if sorter.ndim != 1:
            raise ValueError(f"searchsorted: sorter must be one-dimensional, got ndim={sorter.ndim}")
        sort_idx = sorter.tolist()
        # negative sorter entries would die as OverflowError at the unsigned
        # pyo3 boundary; rstsr validates bounds rust-side
        if builtins.any(i < 0 for i in sort_idx):
            raise ValueError("searchsorted: sorter entries must be non-negative")
    return _wrap(_searchsorted(h1, h2, side == "right", sort_idx))


def nonzero(x, /):
    return tuple(_wrap(h) for h in _nonzero(_handle(x)))


_INT_PROMOTE = {
    "int8": (1, True), "int16": (2, True), "int32": (3, True), "int64": (4, True),
    "uint8": (1, False), "uint16": (2, False), "uint32": (3, False), "uint64": (4, False),
}


def _common_int_dtype(d1, d2, /, opname="isin"):
    """Common integer dtype of two int dtypes (the standard's promotion table).

    rstsr's isin/join kernels are single-dtype, so mixed integer pairs are
    brought to their promoted dtype through the existing astype path
    (value-preserving); mixed non-integer pairs stay declined (G-009).
    """
    try:
        r1, s1 = _INT_PROMOTE[d1.name]
        r2, s2 = _INT_PROMOTE[d2.name]
    except KeyError:
        raise TypeError(
            f"{opname}: mixed-dtype operands {d1.name}/{d2.name} are not provided by "
            f"rstsr (gap G-009); use matching dtypes or cast first"
        ) from None
    if s1 == s2:
        return d1 if r1 >= r2 else d2
    # mixed sign: smallest signed dtype covering both widths; uint64 has no
    # signed partner (int64 cannot hold all uint64 values) — declined
    width = _py_max(r1 if s1 else r1 + 1, r2 if s2 else r2 + 1)
    if width > 4:
        raise TypeError(
            f"{opname}: mixed-dtype operands {d1.name}/{d2.name} have no promotable "
            f"common integer dtype (gap G-009); use matching dtypes or cast first"
        )
    return (int8, int16, int32, int64)[width - 1]


def _promote_pair(d1, d2, /, opname="concat"):
    """The array-API common dtype of two dtypes (bool / int / uint / float /
    complex). Used by the joins: their rstsr kernels are single-dtype, so parts
    are cast to their common dtype first — the same cast-then-apply shape as
    `isin` and the accumulation-dtype path."""
    if d1.name == d2.name:
        return d1
    if d1.name == "bool":
        return d2
    if d2.name == "bool":
        return d1
    k1, k2 = _kind(d1), _kind(d2)
    if k1 == "integral" and k2 == "integral":
        return _common_int_dtype(d1, d2, opname)
    # integral < real floating < complex floating
    lo, hi = (d1, d2) if _KIND_ORDER[k1] <= _KIND_ORDER[k2] else (d2, d1)
    lo_kind, hi_kind = _kind(lo), _kind(hi)
    if hi_kind == "complex floating":
        # complex64 only if both operands are representable in float32
        fits32 = hi.name == "complex64" and (
            (lo_kind == "complex floating" and lo.name == "complex64")
            or (lo_kind == "real floating" and lo.name == "float32")
            or (lo_kind == "integral" and _INT_PROMOTE[lo.name][0] <= 2)
        )
        return complex64 if fits32 else complex128
    if hi_kind == "real floating":
        fits32 = (
            hi.name == "float32"
            and lo_kind == "integral"
            and _INT_PROMOTE[lo.name][0] <= 2
        )
        return float32 if fits32 else float64
    raise TypeError(f"{opname}: cannot promote {d1.name}/{d2.name}")


def _cast_parts(handles, /, opname="concat"):
    """Bring same-device join parts to their common dtype (no-op when they
    already agree), reusing the astype path."""
    dtypes = [h.dtype() for h in handles]
    common = dtypes[0]
    for d in dtypes[1:]:
        common = _promote_pair(common, d, opname)
    if builtins.all(d.name == common.name for d in dtypes):
        return handles
    return [_astype(h, common, True) for h in handles]


def isin(x1, x2, /, *, invert=False):
    if isinstance(x1, Array) and isinstance(x2, Array):
        if x1.dtype is x2.dtype:
            a, b = x1._h, x2._h
        else:
            common = _common_int_dtype(x1.dtype, x2.dtype)
            a, b = astype(x1, common)._h, astype(x2, common)._h
    elif isinstance(x1, Array):
        a, b = x1._h, _scalar_operand(x2, x1.dtype)._h
    elif isinstance(x2, Array):
        a, b = _scalar_operand(x1, x2.dtype)._h, x2._h
    else:
        raise TypeError("isin: at least one of x1, x2 must be an rstsr_faer.api Array")
    return _wrap(_isin(a, b, _py_bool(invert)))


# unique_* return spec namedtuples; the field names are part of the standard.

UniqueAll = collections.namedtuple("unique_all", ["values", "indices", "inverse_indices", "counts"])
UniqueCounts = collections.namedtuple("unique_counts", ["values", "counts"])
UniqueInverse = collections.namedtuple("unique_inverse", ["values", "inverse_indices"])


def unique_values(x, /):
    return _wrap(_unique_values(_handle(x)))


def unique_counts(x, /):
    values, counts = _unique_counts(_handle(x))
    return UniqueCounts(_wrap(values), _wrap(counts))


def unique_inverse(x, /):
    values, inverse = _unique_inverse(_handle(x))
    return UniqueInverse(_wrap(values), _wrap(inverse))


def unique_all(x, /):
    values, indices, inverse, counts = _unique_all(_handle(x))
    return UniqueAll(_wrap(values), _wrap(indices), _wrap(inverse), _wrap(counts))


# --------------------------------------------------------------- data types ---


def astype(x, /, dtype, *, copy=True, device=None):
    _check_device(device)
    return _wrap(_astype(_handle(x), dtype, copy))


def finfo(type, /):
    if isinstance(type, Array):
        type = type.dtype
    # finfo accepts complex dtypes per the spec; report the real component's info
    if type is complex64:
        type = float32
    elif type is complex128:
        type = float64
    return _finfo(type)


def iinfo(type, /):
    if isinstance(type, Array):
        type = type.dtype
    return _iinfo(type)


# ------------------------------------------------------------------- linalg --
# array-API `linalg` extension, over rstsr's existing entries only. Members the
# standard defines but rstsr/faer does not provide (qr, eig, eigvals,
# matrix_norm, matrix_power, matrix_rank, outer, cross, trace, and
# the general-`ord` norms) are absent here — rust-side gaps, recorded in the
# gap register, never stubbed. None of these wrappers batch: rstsr's faer
# factorizations are 2-D only, so a stacked input is passed through and
# declined rust-side.

_EighResult = collections.namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
_SVDResult = collections.namedtuple("SVDResult", ["U", "S", "Vh"])
_SLogDetResult = collections.namedtuple("SLogDetResult", ["sign", "logabsdet"])


def matmul(x1, x2, /):
    return _wrap(_linalg_matmul(_handle(x1), _handle(x2)))


def matrix_transpose(x, /):
    return _wrap(_linalg_matrix_transpose(_handle(x)))


def vecdot(x1, x2, /, *, axis=-1):
    return _wrap(_linalg_vecdot(_handle(x1), _handle(x2), axis))


def tensordot(x1, x2, /, *, axes=2):
    return _wrap(_linalg_tensordot(_handle(x1), _handle(x2), axes))


def cholesky(x, /, *, upper=False):
    return _wrap(_linalg_cholesky(_handle(x), upper))


def det(x, /):
    return _wrap(_linalg_det(_handle(x)))


def diagonal(x, /, *, offset=0):
    return _wrap(_linalg_diagonal(_handle(x), offset))


def eigh(x, /):
    w, v = _linalg_eigh(_handle(x))
    return _EighResult(_wrap(w), _wrap(v))


def eigvalsh(x, /):
    return _wrap(_linalg_eigvalsh(_handle(x)))


def inv(x, /):
    return _wrap(_linalg_inv(_handle(x)))


def pinv(x, /, *, rtol=None):
    return _wrap(_linalg_pinv(_handle(x), rtol))


def solve(x1, x2, /):
    return _wrap(_linalg_solve(_handle(x1), _handle(x2)))


def svd(x, /, *, full_matrices=True):
    u, s, vh = _linalg_svd(_handle(x), _py_bool(full_matrices))
    return _SVDResult(_wrap(u), _wrap(s), _wrap(vh))


def svdvals(x, /):
    return _wrap(_linalg_svdvals(_handle(x)))


def slogdet(x, /):
    sign, logabsdet = _linalg_slogdet(_handle(x))
    return _SLogDetResult(_wrap(sign), _wrap(logabsdet))


class _LinalgNamespace:
    """The array-API ``linalg`` extension namespace (API version 2025.12)."""

    cholesky = staticmethod(cholesky)
    det = staticmethod(det)
    diagonal = staticmethod(diagonal)
    eigh = staticmethod(eigh)
    eigvalsh = staticmethod(eigvalsh)
    inv = staticmethod(inv)
    matmul = staticmethod(matmul)
    matrix_transpose = staticmethod(matrix_transpose)
    pinv = staticmethod(pinv)
    solve = staticmethod(solve)
    svd = staticmethod(svd)
    svdvals = staticmethod(svdvals)
    slogdet = staticmethod(slogdet)
    tensordot = staticmethod(tensordot)
    vecdot = staticmethod(vecdot)


linalg = _LinalgNamespace()


# ------------------------------------------------------ dtype promotion -----
# The standard mandates runtime promotion queries that Rust answers with a
# trait bound (see the module docstring and src/promotion.rs).

_DTYPE_BY_NAME = {
    "bool": bool,
    "int8": int8, "int16": int16, "int32": int32, "int64": int64,
    "uint8": uint8, "uint16": uint16, "uint32": uint32, "uint64": uint64,
    "float32": float32, "float64": float64,
    "complex64": complex64, "complex128": complex128,
}
_ALL_DTYPE_NAMES = tuple(_DTYPE_BY_NAME)

# isdtype kind membership (the seven kind names of the standard)
_KIND_MEMBERS = {
    "bool": ("bool",),
    "signed integer": ("int8", "int16", "int32", "int64"),
    "unsigned integer": ("uint8", "uint16", "uint32", "uint64"),
    "integral": ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"),
    "real floating": ("float32", "float64"),
    "complex floating": ("complex64", "complex128"),
    "numeric": (
        "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
        "float32", "float64", "complex64", "complex128",
    ),
}


def _dtype_name(x, /):
    """Dtype name of an array/dtype argument; None for a Python scalar."""
    if isinstance(x, Array):
        return x.dtype.name
    if isinstance(x, Dtype):
        return x.name
    if isinstance(x, (_py_bool, _py_int, _py_float, _py_complex)):
        return None
    raise TypeError(f"expected an array, a dtype, or a Python scalar, got {type(x).__name__}")


def isdtype(dtype, kind):
    if not isinstance(dtype, Dtype):
        raise TypeError(f"isdtype: dtype must be a dtype object, got {type(dtype).__name__}")
    kinds = kind if isinstance(kind, tuple) else (kind,)
    for k in kinds:
        if isinstance(k, Dtype):
            if dtype is k:
                return True
        elif isinstance(k, str):
            members = _KIND_MEMBERS.get(k)
            if members is None:
                raise ValueError(f"isdtype: unrecognized kind {k!r}")
            if dtype.name in members:
                return True
        else:
            raise TypeError(f"isdtype: kind entries must be str or dtype, got {type(k).__name__}")
    return False


def can_cast(from_, to, /):
    f = _dtype_name(from_)
    if f is None:
        raise TypeError("can_cast: from_ must be an array or a dtype")
    t = _dtype_name(to)
    if t is None:
        raise TypeError("can_cast: to must be a dtype")
    # per the standard, `to` is reachable when some dtype promotes `from_` to it
    return builtins.any(_pkg.dtype_promote(f, o) == t for o in _ALL_DTYPE_NAMES)


def result_type(*arrays_and_dtypes):
    # Python scalars are weak: they never promote the array/dtype arguments.
    names = [n for n in map(_dtype_name, arrays_and_dtypes) if n is not None]
    if not names:
        raise ValueError("result_type: at least one array or dtype is required")
    res = names[0]
    for n in names[1:]:
        res = _pkg.dtype_promote(res, n)
    return _DTYPE_BY_NAME[res]


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
    "empty_like", "zeros_like", "ones_like", "full_like",
    "eye", "linspace", "tril", "triu", "meshgrid",
    # elementwise
    "add", "subtract", "multiply", "divide", "negative", "abs", "positive",
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "ceil",
    "conj", "copysign", "cos", "cosh", "exp", "expm1", "floor", "floor_divide",
    "hypot", "imag", "log", "log1p", "log2", "log10", "logaddexp", "maximum",
    "minimum", "clip", "nextafter", "pow", "real", "reciprocal", "remainder", "round",
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
    "reshape", "permute_dims", "broadcast_arrays", "broadcast_shapes",
    "concat", "stack", "unstack", "expand_dims", "squeeze", "flip", "moveaxis",
    "repeat", "roll", "tile", "diff",
    # linear algebra
    "matmul", "matrix_transpose", "vecdot", "tensordot", "linalg",
    # sorting
    "sort", "argsort",
    # searching / indexing
    "argmax", "argmin", "count_nonzero", "take", "where",
    "searchsorted", "nonzero", "take_along_axis",
    # set functions
    "isin", "unique_values", "unique_counts", "unique_inverse", "unique_all",
    # data types
    "astype", "finfo", "iinfo", "can_cast", "isdtype", "result_type",
    # constants / sentinels
    "newaxis",
]
