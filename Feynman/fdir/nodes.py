from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from .types import TensorType, Shape, DType, Layout

@dataclass
class Propagator:
    """Internal or external line carrying typed state across vertices."""
    id: str
    src_vertex_id: str
    dst_vertex_id: str
    tensor_type: TensorType
    label: str = ""


class Vertex:
    """Base interaction vertex in the Feynman Diagrammatic IR."""
    def __init__(self, id: str, op_type: str,
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None,
                 attributes: Optional[Dict[str, Any]] = None):
        self.id = id
        self.op_type = op_type
        self.inputs: List[str] = inputs or []
        self.outputs: List[str] = outputs or []
        self.attributes: Dict[str, Any] = attributes or {}

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        raise NotImplementedError(f"get_output_type not implemented for {self.__class__.__name__}")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id!r} op={self.op_type!r}>"


class InputVertex(Vertex):
    """External incoming particle/token state."""
    def __init__(self, id: str, tensor_type: TensorType):
        super().__init__(id=id, op_type="Input", attributes={"tensor_type": tensor_type})

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        return self.attributes["tensor_type"]


class OutputVertex(Vertex):
    """External outgoing token state."""
    def __init__(self, id: str):
        super().__init__(id=id, op_type="Output")

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        if not input_types:
            raise ValueError(f"OutputVertex {self.id} received no inputs")
        return input_types[0]


class ContractionVertex(Vertex):
    """Rank contraction vertex (MatMul / Linear Projection).

    Implements the Feynman rule: contraction over a shared index k.
      [..., M, K] × [..., K, N] → [..., M, N]

    The contracted dimension is validated during get_output_type.
    """
    def __init__(self, id: str, transpose_b: bool = False):
        super().__init__(
            id=id,
            op_type="Contraction",
            attributes={"transpose_b": transpose_b}
        )

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        if len(input_types) < 2:
            raise ValueError(f"ContractionVertex {self.id} requires at least 2 inputs, got {len(input_types)}")
        t1, t2 = input_types[0], input_types[1]
        shape1, shape2 = list(t1.shape.dims), list(t2.shape.dims)

        transpose_b = self.attributes.get("transpose_b", False)
        if transpose_b and len(shape2) >= 2:
            shape2[-2], shape2[-1] = shape2[-1], shape2[-2]

        # Contraction axis conservation law: shape1[-1] must equal shape2[-2]
        k1 = shape1[-1] if shape1 else None
        k2 = shape2[-2] if len(shape2) >= 2 else (shape2[-1] if shape2 else None)

        if k1 is not None and k2 is not None:
            if isinstance(k1, int) and isinstance(k2, int) and k1 != k2:
                raise ValueError(
                    f"ContractionVertex {self.id}: contraction axis mismatch: "
                    f"lhs last dim = {k1}, rhs second-to-last dim = {k2}"
                )
            if isinstance(k1, str) and isinstance(k2, str) and k1 != k2:
                raise ValueError(
                    f"ContractionVertex {self.id}: contraction axis symbol mismatch: "
                    f"lhs = '{k1}', rhs = '{k2}'"
                )

        # Output shape: broadcast batch dims + (M, N)
        # For [..., M, K] × [K, N] → [..., M, N]  (standard linear projection)
        # For [..., M, K] × [..., K, N] → [..., M, N]  (batched matmul)
        batch1 = shape1[:-2] if len(shape1) >= 2 else ()
        batch2 = shape2[:-2] if len(shape2) >= 2 else ()

        # Simple broadcast: use the longer batch prefix
        batch_dims = tuple(batch1 if len(batch1) >= len(batch2) else batch2)

        m = shape1[-2] if len(shape1) >= 2 else 1
        n = shape2[-1] if shape2 else 1

        out_shape = Shape(batch_dims + (m, n))
        return TensorType(shape=out_shape, dtype=t1.dtype, layout=t1.layout)


class AttentionVertex(Vertex):
    """Rank-4 SDPA interaction vertex: Softmax(Q K^T / sqrt(d_k)) V."""
    def __init__(self, id: str, num_heads: int, head_dim: int):
        super().__init__(
            id=id,
            op_type="Attention",
            attributes={"num_heads": num_heads, "head_dim": head_dim}
        )

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        if len(input_types) < 3:
            raise ValueError(f"AttentionVertex {self.id} requires Q, K, V (3 inputs), got {len(input_types)}")
        return input_types[0]  # Output shape matches Q


class PointwiseVertex(Vertex):
    """Pointwise activation or binary elementwise vertex."""
    UNARY_OPS = {"ReLU", "GELU", "Softmax", "Sigmoid", "Tanh"}
    BINARY_OPS = {"Add", "Mul", "Sub"}

    def __init__(self, id: str, sub_op: str):
        super().__init__(id=id, op_type="Pointwise", attributes={"sub_op": sub_op})

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        sub_op = self.attributes.get("sub_op", "Add")
        if sub_op in self.BINARY_OPS and len(input_types) < 2:
            raise ValueError(f"PointwiseVertex {self.id} ({sub_op}) requires 2 inputs, got {len(input_types)}")
        return input_types[0]


class ResidualVertex(Vertex):
    """Residual bypass addition vertex (y = x + F(x))."""
    def __init__(self, id: str):
        super().__init__(id=id, op_type="ResidualAdd")

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        if len(input_types) < 2:
            raise ValueError(f"ResidualVertex {self.id} requires 2 inputs (bypass + branch), got {len(input_types)}")
        return input_types[0]


class NormVertex(Vertex):
    """Normalization vertex (LayerNorm / RMSNorm)."""
    def __init__(self, id: str, norm_type: str = "LayerNorm", eps: float = 1e-5):
        super().__init__(id=id, op_type="Norm", attributes={"norm_type": norm_type, "eps": eps})

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        if not input_types:
            raise ValueError(f"NormVertex {self.id} requires at least 1 input")
        return input_types[0]


class TransposeVertex(Vertex):
    """Permute dimension vertex."""
    def __init__(self, id: str, perm: Tuple[int, ...]):
        super().__init__(id=id, op_type="Transpose", attributes={"perm": perm})

    def get_output_type(self, input_types: List[TensorType]) -> TensorType:
        if not input_types:
            raise ValueError(f"TransposeVertex {self.id} requires at least 1 input")
        in_shape = input_types[0].shape.dims
        perm = self.attributes["perm"]
        if len(perm) != len(in_shape):
            raise ValueError(
                f"TransposeVertex {self.id}: perm length {len(perm)} != shape rank {len(in_shape)}"
            )
        out_dims = tuple(in_shape[i] for i in perm)
        return TensorType(shape=Shape(out_dims), dtype=input_types[0].dtype, layout=input_types[0].layout)
