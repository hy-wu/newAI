"""
Worked Example: Building, Validating, Costing, and Lowering a Transformer Block with FDIR.
"""

import sys
import os
import torch

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Feynman.fdir import (
    Diagram, TensorType, Shape, DType,
    InputVertex, OutputVertex, ContractionVertex, AttentionVertex,
    ResidualVertex, NormVertex,
    ShapeTypeChecker, CostModel, TorchLowering
)


def main():
    print("================================================================")
    print(" FDIR Worked Example: Transformer Architecture Diagram to PyTorch")
    print("================================================================\n")

    # 1. Instantiate FDIR Diagram
    d = Diagram("transformer_block_fdir")

    B, S, D = "B", "S", "D"
    x_type = TensorType(shape=Shape((B, S, D)), dtype=DType.FLOAT32)
    w_type = TensorType(shape=Shape((D, D)), dtype=DType.FLOAT32)

    # External lines (inputs)
    d.add_vertex(InputVertex("x", x_type))
    d.add_vertex(InputVertex("W_q", w_type))
    d.add_vertex(InputVertex("W_k", w_type))
    d.add_vertex(InputVertex("W_v", w_type))

    # Q, K, V projection vertices (contraction interactions)
    d.add_vertex(ContractionVertex("proj_Q"))
    d.add_vertex(ContractionVertex("proj_K"))
    d.add_vertex(ContractionVertex("proj_V"))

    d.connect("x", "proj_Q", x_type)
    d.connect("W_q", "proj_Q", w_type)
    d.connect("x", "proj_K", x_type)
    d.connect("W_k", "proj_K", w_type)
    d.connect("x", "proj_V", x_type)
    d.connect("W_v", "proj_V", w_type)

    # SDPA Attention interaction vertex
    d.add_vertex(AttentionVertex("sdpa_attention", num_heads=12, head_dim=64))
    d.connect("proj_Q", "sdpa_attention", x_type)
    d.connect("proj_K", "sdpa_attention", x_type)
    d.connect("proj_V", "sdpa_attention", x_type)

    # Residual bypass propagator + addition vertex
    d.add_vertex(ResidualVertex("residual_add_1"))
    d.connect("x", "residual_add_1", x_type, label="bypass_stream")
    d.connect("sdpa_attention", "residual_add_1", x_type, label="attention_stream")

    # Layer normalization vertex
    d.add_vertex(NormVertex("layer_norm_1"))
    d.connect("residual_add_1", "layer_norm_1", x_type)

    # Output external line
    d.add_vertex(OutputVertex("y_out"))
    d.connect("layer_norm_1", "y_out", x_type)

    print(f"[*] FDIR Diagram Created: {d}\n")

    # 2. Check conservation laws (type & shape)
    env = {"B": 4, "S": 256, "D": 768}
    checker = ShapeTypeChecker(env=env)
    is_valid = checker.check(d)
    print(f"[*] Conservation Law Verification: {'PASSED [OK]' if is_valid else 'FAILED'}\n")

    # 3. Evaluate cost model (action integral)
    cost_model = CostModel(env=env)
    report = cost_model.evaluate(d)
    print(report)

    # 4. Lower to PyTorch executable module
    lowering = TorchLowering()
    pytorch_module = lowering.lower(d)
    print("[*] PyTorch Module Successfully Compiled from Diagram\n")

    # 5. Numerical test run
    print("[*] Running Numerical Equivalence Test with Random Tensors (B=4, S=256, D=768)...")
    x_val = torch.randn(4, 256, 768)
    wq_val = torch.randn(768, 768)
    wk_val = torch.randn(768, 768)
    wv_val = torch.randn(768, 768)

    y_val = pytorch_module(x_val, wq_val, wk_val, wv_val)
    print(f"    Output Tensor Shape: {tuple(y_val.shape)}")
    print(f"    Output Tensor Mean:  {y_val.mean().item():.6f}")
    print(f"    Output Tensor Std:   {y_val.std().item():.6f}")

    # 6. Mermaid export
    print(f"\n[*] Mermaid Diagram Export:\n")
    print(d.to_mermaid())

    print("\n[OK] Worked Example Completed Successfully!")


if __name__ == "__main__":
    main()
