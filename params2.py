
import torch
# from KD_models_nodil_4B_1024 import UNetGenerator_T
from KD_st_noMSA_1conv_lay123_4pool import UNetGenerator

from calflops import calculate_flops
from ptflops import get_model_complexity_info
from thop import profile



print('Initializing model...')

model = UNetGenerator().cuda()  # student model
# model = UNet(n_channels=3, n_classes=3).cuda()

# 모델을 평가 모드로 전환
model.eval()

# 입력 크기 정의 (배치 크기 포함하지 않음)
inputsize = (1, 4, 224, 224)  # 배치 크기 1, 채널 4, 224x224 이미지


# FLOPs 계산
flops, macs, params = calculate_flops(
    model=model,
    input_shape=inputsize,  # 두 개의 입력 전달
    output_as_string=True,
    output_precision=4
)

print("==============calflops================")
print("GatedGenerator FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))



print("==============inference time_sol_1================")

import torch

model.eval()
model = model.cuda()

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)

dummy_input = torch.randn(inputsize).cuda()

with torch.no_grad():  # Gradient 연산 비활성화
    starter.record()
output = model(dummy_input)
ender.record()

torch.cuda.synchronize()  # Gpu 연산이 완료될 때까지 대기

infer_time = starter.elapsed_time(ender)
print("Elapsed time: {} ms".format(infer_time))  # milliseconds

print("==============inference time_sol_2================")
import time

def measure_inference_time(model, input_tensor, repeat=100):
    torch.cuda.empty_cache()
    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_tensor)

        torch.cuda.synchronize()
        start = time.time()
        for _ in range(repeat):
            _ = model(input_tensor)
        torch.cuda.synchronize()
        end = time.time()

        avg_time = (end - start) * 1000 / repeat  # ms
        return avg_time

inference_time = measure_inference_time(model, dummy_input)
print(f"Inference Time (ms): {inference_time:.4f}")

print("==============memory usage_sol_1================")

def get_gpu_memory_usage():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_input)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

memory_usage = get_gpu_memory_usage()
print(f"Memory Usage (MB): {memory_usage:.4f}")





#
# print("==============ptflops================")
# input_size = (4, 224, 224)
# with torch.cuda.device(0):
#     macs, params = get_model_complexity_info(model, input_size, as_strings=True,
#                                              print_per_layer_stat=False, verbose=False)
#     print(f"FLOPs: {macs} | Params: {params}")
#
# print("==============thop================")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dummy_input = torch.randn(1, *input_size).to(device)
# macs, params = profile(model, inputs=(dummy_input,), verbose=False)
# print(macs)
# print(f"FLOPs: {macs / 1e9:.4f} GFLOPs | Params: {params / 1e6:.4f} M")

# # 모델 요약
# summary(model, input_size=input_size)




