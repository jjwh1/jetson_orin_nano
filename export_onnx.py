import torch
from KD_st_noMSA_1conv_lay123_4pool import UNetGenerator

# 1. 모델 로드
model = UNetGenerator().cuda().eval()

# 2. 더미 입력 (고정 shape)
dummy_input = torch.randn(1, 4, 224, 224).cuda()

# 3. ONNX export
torch.onnx.export(
    model,
    dummy_input,
    "unet.onnx",
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None,          # TensorRT용 → static
    do_constant_folding=True
)

print("✅ ONNX export done")
