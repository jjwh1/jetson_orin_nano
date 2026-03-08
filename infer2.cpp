#include <NvInfer.h>
#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TRT] " << msg << std::endl;
        }
    }
};

static Logger gLogger;

template <typename T>
struct TRTDestroy {
    void operator()(T* obj) const {
        delete obj;
    }
};

size_t getElementSize(nvinfer1::DataType dtype) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kBOOL:  return 1;
#if NV_TENSORRT_MAJOR >= 10
        case nvinfer1::DataType::kUINT8: return 1;
#endif
        default:
            throw std::runtime_error("Unsupported TensorRT datatype");
    }
}

std::vector<char> loadFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);

    if (!file) {
        throw std::runtime_error("Failed to read engine file: " + path);
    }
    return buffer;
}

int64_t volume(const nvinfer1::Dims& dims) {
    int64_t v = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] < 0) return -1;  // dynamic dimension remains
        v *= dims.d[i];
    }
    return v;
}

std::string dimsToString(const nvinfer1::Dims& dims) {
    std::string s = "[";
    for (int i = 0; i < dims.nbDims; ++i) {
        s += std::to_string(dims.d[i]);
        if (i + 1 < dims.nbDims) s += ", ";
    }
    s += "]";
    return s;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 6) {
            std::cerr << "Usage:\n";
            std::cerr << "  " << argv[0] << " engine.plan\n";
            std::cerr << "  " << argv[0] << " engine.plan N C H W   (for dynamic input)\n";
            return 1;
        }

        const std::string enginePath = argv[1];

        auto engineData = loadFile(enginePath);

        std::unique_ptr<nvinfer1::IRuntime, TRTDestroy<nvinfer1::IRuntime>> runtime(
            nvinfer1::createInferRuntime(gLogger));
        if (!runtime) throw std::runtime_error("Failed to create TensorRT runtime");

        std::unique_ptr<nvinfer1::ICudaEngine, TRTDestroy<nvinfer1::ICudaEngine>> engine(
            runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
        if (!engine) throw std::runtime_error("Failed to deserialize engine");

        std::unique_ptr<nvinfer1::IExecutionContext, TRTDestroy<nvinfer1::IExecutionContext>> context(
            engine->createExecutionContext());
        if (!context) throw std::runtime_error("Failed to create execution context");

        int32_t numIOTensors = engine->getNbIOTensors();
        std::cout << "Number of IO tensors: " << numIOTensors << "\n";

        std::string inputName, outputName;
        for (int i = 0; i < numIOTensors; ++i) {
            const char* tensorName = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(tensorName);
            auto dtype = engine->getTensorDataType(tensorName);
            auto dims = engine->getTensorShape(tensorName);

            std::cout << (mode == nvinfer1::TensorIOMode::kINPUT ? "[INPUT] " : "[OUTPUT] ")
                      << tensorName
                      << " dtype=" << static_cast<int>(dtype)
                      << " shape=" << dimsToString(dims) << "\n";

            if (mode == nvinfer1::TensorIOMode::kINPUT && inputName.empty()) {
                inputName = tensorName;
            }
            if (mode == nvinfer1::TensorIOMode::kOUTPUT && outputName.empty()) {
                outputName = tensorName;
            }
        }

        if (inputName.empty() || outputName.empty()) {
            throw std::runtime_error("Could not find input/output tensor names");
        }

        nvinfer1::Dims inputDims = engine->getTensorShape(inputName.c_str());

        bool hasDynamic = false;
        for (int i = 0; i < inputDims.nbDims; ++i) {
            if (inputDims.d[i] < 0) {
                hasDynamic = true;
                break;
            }
        }

        if (hasDynamic) {
            if (argc != 6) {
                throw std::runtime_error(
                    "Dynamic input detected. Please provide N C H W.\n"
                    "Example: ./infer_trt10 unet_fp16.plan 1 1 224 224");
            }

            nvinfer1::Dims4 actualInputDims{
                std::stoi(argv[2]),
                std::stoi(argv[3]),
                std::stoi(argv[4]),
                std::stoi(argv[5])
            };

            bool ok = context->setInputShape(inputName.c_str(), actualInputDims);
            if (!ok) {
                throw std::runtime_error("Failed to set dynamic input shape");
            }
            inputDims = context->getTensorShape(inputName.c_str());
        } else {
            // even for static shape, context shape is safer
            inputDims = context->getTensorShape(inputName.c_str());
        }

        nvinfer1::Dims outputDims = context->getTensorShape(outputName.c_str());

        std::cout << "Resolved input shape : " << dimsToString(inputDims) << "\n";
        std::cout << "Resolved output shape: " << dimsToString(outputDims) << "\n";

        int64_t inputVol = volume(inputDims);
        int64_t outputVol = volume(outputDims);
        if (inputVol <= 0 || outputVol <= 0) {
            throw std::runtime_error("Failed to resolve tensor shapes completely");
        }

        auto inputType = engine->getTensorDataType(inputName.c_str());
        auto outputType = engine->getTensorDataType(outputName.c_str());

        size_t inputBytes = static_cast<size_t>(inputVol) * getElementSize(inputType);
        size_t outputBytes = static_cast<size_t>(outputVol) * getElementSize(outputType);

        std::cout << "Input bytes : " << inputBytes << "\n";
        std::cout << "Output bytes: " << outputBytes << "\n";

        void* dInput = nullptr;
        void* dOutput = nullptr;
        cudaMalloc(&dInput, inputBytes);
        cudaMalloc(&dOutput, outputBytes);

        std::vector<uint8_t> hInput(inputBytes, 0);
        std::vector<uint8_t> hOutput(outputBytes, 0);

        // 간단히 랜덤 입력 채우기
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> dist(0, 255);
        for (size_t i = 0; i < hInput.size(); ++i) {
            hInput[i] = static_cast<uint8_t>(dist(rng));
        }

        cudaMemcpy(dInput, hInput.data(), inputBytes, cudaMemcpyHostToDevice);

        bool ok1 = context->setTensorAddress(inputName.c_str(), dInput);
        bool ok2 = context->setTensorAddress(outputName.c_str(), dOutput);
        if (!ok1 || !ok2) {
            throw std::runtime_error("Failed to set tensor addresses");
        }

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // warm-up
        for (int i = 0; i < 20; ++i) {
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("Warmup inference failed");
            }
        }
        cudaStreamSynchronize(stream);

        const int numRuns = 100;
        float totalMs = 0.0f;

        for (int i = 0; i < numRuns; ++i) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);

            cudaEventRecord(start, stream);
            if (!context->enqueueV3(stream)) {
                throw std::runtime_error("Inference failed");
            }
            cudaEventRecord(stop, stream);
            cudaEventSynchronize(stop);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start, stop);
            totalMs += ms;

            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        cudaMemcpyAsync(hOutput.data(), dOutput, outputBytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::cout << "Average inference time: " << (totalMs / numRuns) << " ms\n";
        std::cout << "FPS: " << (1000.0f / (totalMs / numRuns)) << "\n";

        // 출력 일부만 확인
        std::cout << "Output first 16 bytes: ";
        for (int i = 0; i < std::min<size_t>(16, hOutput.size()); ++i) {
            std::cout << static_cast<int>(hOutput[i]) << " ";
        }
        std::cout << "\n";

        cudaStreamDestroy(stream);
        cudaFree(dInput);
        cudaFree(dOutput);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
}
