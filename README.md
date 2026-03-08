# TensorRT_KDISR

TensorRT inference pipeline for the **KDISR UNet generator model**.

This repository provides a simple workflow to:

- Export a PyTorch model to ONNX
- Convert ONNX to a TensorRT engine
- Benchmark inference latency using TensorRT

Designed for **Jetson / CUDA GPU environments**.

---

# Repository Structure

```
TensorRT_KDISR
│
├── KD_st_noMSA_1conv_lay123_4pool.py   # UNet generator model definition
├── export_onnx.py                      # PyTorch → ONNX export script
├── trt_infer_time.py                   # TensorRT inference latency measurement
├── params2.py                          # Model configuration
│
├── unet.onnx                           # Raw ONNX model
├── unet_sim.onnx                       # Simplified ONNX model
│
├── unet_fp16.plan                      # TensorRT FP16 engine
├── unet_fp32.plan                      # TensorRT FP32 engine
├── unet_int8.plan                      # TensorRT INT8 engine
│
└── README.md
```

---

# Pipeline

```
PyTorch Model
      ↓
ONNX Export
      ↓
ONNX Simplification
      ↓
TensorRT Engine Build
      ↓
TensorRT Inference Benchmark
```

---

# 1. Export PyTorch Model to ONNX

Run the ONNX export script:

```
python export_onnx.py
```

This generates:

```
unet.onnx
```

Input shape used for export:

```
(1, 4, 224, 224)
```

Static shapes are used for optimal TensorRT performance.

---

# 2. Simplify ONNX Model (Recommended)

Simplify the ONNX graph before TensorRT conversion:

```
onnxsim unet.onnx unet_sim.onnx
```

This produces:

```
unet_sim.onnx
```

---

# 3. Build TensorRT Engine

Use TensorRT's `trtexec` tool.

Example for FP16:

```
trtexec \
--onnx=unet_sim.onnx \
--saveEngine=unet_fp16.plan \
--fp16
```

Output:

```
unet_fp16.plan
```

Other precision modes:

| Precision | Option |
|----------|--------|
| FP32 | default |
| FP16 | `--fp16` |
| INT8 | `--int8` (requires calibration) |

---

# 4. TensorRT Inference Benchmark

Run:

```
python trt_infer_time.py
```

Example output:

```
TensorRT FP16 Inference Time: 2.10 ms
```

The script performs:

- GPU memory allocation
- Warm-up inference
- Multiple inference iterations
- Average latency measurement

---

# Requirements

Python packages:

```
pip install onnx onnxsim pycuda
```

Required software:

- CUDA
- TensorRT
- PyTorch

TensorRT Python wheel must match your CUDA version.

---

# Notes

- Static input shape is used for best TensorRT optimization.
- Intended for **Jetson GPU inference benchmarking**.
- TensorRT engines are generated using `trtexec`.

---

# License

For research and experimental use.
