import torch
from models.experimental import attempt_load
from pathlib import Path

# Load model
weights = "models/best_model_1.3.pt"
device = "cpu"
model = attempt_load(weights, device=device)
model.eval()

# Verify class names
print("Classes:", model.names)  # should be 4 classes

# Export to ONNX
dummy_input = torch.randn(1, 3, 640, 640)
onnx_path = Path(weights).with_suffix(".onnx")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["images"],
    output_names=["output"],
    opset_version=12,
    dynamic_axes={
        "images": {0: "batch"},
        "output": {0: "batch"}
    }
)

print(f"Exported to {onnx_path}")
