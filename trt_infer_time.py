import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# =========================
# 1. Engine load
# =========================
with open("unet_int8.plan", "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# =========================
# 2. Tensor names
# =========================
input_name = engine.get_tensor_name(0)
output_name = engine.get_tensor_name(1)

print("Input tensor:", input_name)
print("Output tensor:", output_name)

# =========================
# 3. Input shape
# =========================
input_shape = (1, 4, 224, 224)
context.set_input_shape(input_name, input_shape)

# (TensorRT 8.6용 확인)
assert tuple(context.get_tensor_shape(input_name)) == input_shape

# =========================
# 4. Buffer allocation
# =========================
input_size = int(np.prod(input_shape) * np.float16().itemsize)
output_shape = tuple(context.get_tensor_shape(output_name))
output_size = int(np.prod(output_shape) * np.float16().itemsize)

d_input = cuda.mem_alloc(input_size)
d_output = cuda.mem_alloc(output_size)

context.set_tensor_address(input_name, int(d_input))
context.set_tensor_address(output_name, int(d_output))

stream = cuda.Stream()
dummy_input = np.random.randn(*input_shape).astype(np.float16)

# =========================
# 5. Warm-up
# =========================
for _ in range(20):
    cuda.memcpy_htod_async(d_input, dummy_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)
stream.synchronize()

# =========================
# 6. Timing
# =========================
repeat = 1000
start = time.time()

for _ in range(repeat):
    cuda.memcpy_htod_async(d_input, dummy_input, stream)
    context.execute_async_v3(stream_handle=stream.handle)

stream.synchronize()
end = time.time()

print(f"🚀 TensorRT FP16 Inference Time: {(end-start)*1000/repeat:.4f} ms")
