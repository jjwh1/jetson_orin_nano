#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_fp16.h>

using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

int main() {

    std::string engine_file = "unet_fp16.plan";

    // =========================
    // 1. Engine load
    // =========================
    std::ifstream file(engine_file, std::ios::binary);
    file.seekg(0, std::ifstream::end);
    size_t size = file.tellg();
    file.seekg(0, std::ifstream::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    file.close();

    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
    IExecutionContext* context = engine->createExecutionContext();

    // =========================
    // 2. Tensor names
    // =========================
    const char* input_name = engine->getTensorName(0);
    const char* output_name = engine->getTensorName(1);

    std::cout << "Input tensor: " << input_name << std::endl;
    std::cout << "Output tensor: " << output_name << std::endl;

    // =========================
    // 3. Input shape
    // =========================
    int N = 1, C = 4, H = 224, W = 224;
    context->setInputShape(input_name, Dims4(N, C, H, W));

    // =========================
    // 4. Buffer allocation
    // =========================
    size_t input_size = N*C*H*W*sizeof(__half);
    size_t output_size = N*C*H*W*sizeof(__half);

    void* d_input;
    void* d_output;

    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);

    context->setTensorAddress(input_name, d_input);
    context->setTensorAddress(output_name, d_output);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<__half> dummy_input(N*C*H*W);

    // =========================
    // 5. Warm-up
    // =========================
    for (int i=0; i<20; i++) {

        cudaMemcpyAsync(
            d_input,
            dummy_input.data(),
            input_size,
            cudaMemcpyHostToDevice,
            stream
        );

        context->enqueueV3(stream);
    }

    cudaStreamSynchronize(stream);

    // =========================
    // 6. Timing
    // =========================
    int repeat = 1000;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<repeat; i++) {

        cudaMemcpyAsync(
            d_input,
            dummy_input.data(),
            input_size,
            cudaMemcpyHostToDevice,
            stream
        );

        context->enqueueV3(stream);
    }

    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();

    double latency =
        std::chrono::duration<double, std::milli>(end-start).count() / repeat;

    std::cout << "🚀 TensorRT FP16 Inference Time: "
              << latency << " ms" << std::endl;

    // cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    context->destroy();
    engine->destroy();
    runtime->destroy();

}
