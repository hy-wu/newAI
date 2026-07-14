from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Union, Dict, Optional

class DType(Enum):
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT64 = "int64"

    def bytes_per_element(self) -> int:
        _table = {
            DType.FLOAT32: 4, DType.INT32: 4,
            DType.FLOAT16: 2, DType.BFLOAT16: 2,
            DType.INT64: 8,
        }
        return _table.get(self, 4)

class Layout(Enum):
    ROW_MAJOR = "ROW_MAJOR"
    COL_MAJOR = "COL_MAJOR"
    NCHW = "NCHW"
    NHWC = "NHWC"
    SPARSE_COO = "SPARSE_COO"

DimType = Union[int, str]

@dataclass(frozen=True)
class Shape:
    """Immutable tensor shape with support for symbolic (str) and concrete (int) dimensions."""
    dims: Tuple[DimType, ...]

    @property
    def rank(self) -> int:
        return len(self.dims)

    def resolve_dim(self, d: DimType, env: Optional[Dict[str, int]] = None) -> Optional[int]:
        """Resolve a dimension to a concrete int, or None if unresolvable."""
        if isinstance(d, int):
            return d
        if env and d in env:
            return env[d]
        return None

    def num_elements(self, env: Optional[Dict[str, int]] = None) -> int:
        """Compute total element count. Unresolvable symbolic dims default to 1."""
        count = 1
        for d in self.dims:
            resolved = self.resolve_dim(d, env)
            count *= resolved if resolved is not None else 1
        return count

    def is_compatible(self, other: Shape, env: Optional[Dict[str, int]] = None) -> bool:
        """Check shape compatibility enforcing conservation laws.

        Rules:
        - Ranks must match.
        - int vs int: must be equal.
        - str vs str: must be the SAME symbol name (representing the same
          abstract dimension). Different names are incompatible unless env
          resolves them to the same concrete value.
        - int vs str: compatible only if env resolves the str to that int.
        """
        if self.rank != other.rank:
            return False
        for d1, d2 in zip(self.dims, other.dims):
            r1 = self.resolve_dim(d1, env)
            r2 = self.resolve_dim(d2, env)
            # Both concrete: must match
            if r1 is not None and r2 is not None:
                if r1 != r2:
                    return False
            # Both symbolic and unresolved: names must match
            elif isinstance(d1, str) and isinstance(d2, str):
                if d1 != d2:
                    return False
            # One resolved, one not: can't verify — treat as incompatible
            # (conservative: better to reject than to silently pass)
            elif (r1 is None) != (r2 is None):
                return False
        return True

    def __str__(self) -> str:
        return "(" + ", ".join(str(d) for d in self.dims) + ")"

@dataclass
class TensorType:
    """Typed tensor descriptor with shape, dtype, layout, and optional Einstein indices."""
    shape: Shape
    dtype: DType = DType.FLOAT32
    layout: Layout = Layout.ROW_MAJOR
    indices: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def rank(self) -> int:
        return self.shape.rank

    def size_in_bytes(self, env: Optional[Dict[str, int]] = None) -> int:
        return self.shape.num_elements(env) * self.dtype.bytes_per_element()

    def __str__(self) -> str:
        idx_str = f"[{','.join(self.indices)}]" if self.indices else ""
        return f"Tensor{self.shape}{idx_str}:{self.dtype.value}:{self.layout.value}"
