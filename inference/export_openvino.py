import os
import torch
import openvino as ov
import nncf
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.vla_policy import get_policy

def export_to_openvino(output_dir="inference/ir_model"):
    """
    Exports the PyTorch VLA policy to OpenVINO IR format.
    Applies INT8 Quantization using NNCF for optimal Intel Core Ultra NPU/iGPU execution.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading PyTorch VLA Policy...")
    model = get_policy()
    model.eval()

    # Create dummy inputs for tracing
    print("Tracing model for FP16/FP32 export...")
    dummy_pixels = torch.randn(1, 3, 480, 640)
    dummy_language = torch.randn(1, 768)
    
    # Convert to OpenVINO Model
    ov_model = ov.convert_model(
        model, 
        example_input=(dummy_pixels, dummy_language),
        input=[(1, 3, 480, 640), (1, 768)]
    )

    # Save the base IR model (FP16/32)
    ir_path = os.path.join(output_dir, "vla_policy.xml")
    ov.save_model(ov_model, ir_path)
    print(f"Successfully saved base OpenVINO IR to {ir_path}")
    
    # --- UPGRADE: NNCF INT8 Quantization ---
    print("\nApplying NNCF INT8 Post-Training Quantization (PTQ)...")
    
    # Create a dummy calibration dataset for PTQ
    # In a real scenario, this should yield actual representative frames and text embeddings
    def transform_fn(data_item):
        return (torch.randn(1, 3, 480, 640), torch.randn(1, 768))

    # We use 10 dummy samples for calibration
    calibration_dataset = nncf.Dataset([1] * 10, transform_fn)
    
    # Quantize the model
    quantized_model = nncf.quantize(ov_model, calibration_dataset)
    
    # Save the quantized model
    quantized_ir_path = os.path.join(output_dir, "vla_policy_int8.xml")
    ov.save_model(quantized_model, quantized_ir_path)
    print(f"Successfully saved INT8 Quantized OpenVINO IR to {quantized_ir_path}")
    print("Use the INT8 model for maximum throughput on Intel NPU!")

if __name__ == "__main__":
    export_to_openvino()
