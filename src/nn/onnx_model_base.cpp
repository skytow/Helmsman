#include "nn/onnx_model_base.h"

#include <iostream>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <onnxruntime/onnxruntime_c_api.h>

#include "constants.h"
#include "utils/common.h"

OnnxModelBase::OnnxModelBase(const char* modelPath, const char* logid, const char* provider)
    : modelPath_(modelPath)
{
    // Initialize the ONNX Runtime environment
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, logid);
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();

    // Get list of available providers from the runtime
    std::vector<std::string> availableProviders = Ort::GetAvailableProviders();
    auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), OnnxProviders::CUDA);
    OrtCUDAProviderOptions cudaOption;

    std::string providerStr(provider);
    if (providerStr == OnnxProviders::CUDA) {
        if (cudaAvailable == availableProviders.end()) {
            std::cout << "CUDA is not supported by your ONNXRuntime build. Falling back to CPU." << std::endl;
            // CPU is default so we do nothing
        } else {
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
        }
    }
    else if (providerStr == OnnxProviders::CPU || providerStr == OnnxProviders::CPUExecutionProvider) {
        // CPU is the default provider in ONNX Runtime, so no need to append an execution provider.
        // Do nothing.
    }
    else {
        throw std::runtime_error("NotImplemented provider=" + providerStr);
    }

    std::cout << "Inference device: " << providerStr << std::endl;

    #ifdef _WIN32
        auto modelPathW = get_win_path(modelPath);  // For Windows: convert to wstring
        session = Ort::Session(env, modelPathW.c_str(), sessionOptions);
    #else
        session = Ort::Session(env, modelPath, sessionOptions);  // For Linux (and macOS)
    #endif

    // ----------------
    // Initialize input names and copy them to member storage to extend lifetime
    Ort::AllocatorWithDefaultOptions allocator;
    size_t inputNodesNum = session.GetInputCount();
    for (size_t i = 0; i < inputNodesNum; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        inputNodeNames.push_back(std::string(input_name.get()));
    }
    for (const auto& name : inputNodeNames) {
        inputNamesCStr.push_back(name.c_str());
    }

    // -----------------
    // Initialize output names and copy them to member storage
    Ort::AllocatorWithDefaultOptions output_names_allocator;
    size_t outputNodesNum = session.GetOutputCount();
    for (size_t i = 0; i < outputNodesNum; i++) {
        auto output_name = session.GetOutputNameAllocated(i, output_names_allocator);
        outputNodeNames.push_back(std::string(output_name.get()));
    }
    for (const auto& name : outputNodeNames) {
        outputNamesCStr.push_back(name.c_str());
    }

    // -------------------------
    // Initialize model metadata
    model_metadata = session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions metadata_allocator;
    std::vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = model_metadata.GetCustomMetadataMapKeysAllocated(metadata_allocator);
    std::vector<std::string> metadata_keys;
    metadata_keys.reserve(metadataAllocatedKeys.size());
    for (const Ort::AllocatedStringPtr& allocatedString : metadataAllocatedKeys) {
        metadata_keys.emplace_back(allocatedString.get());
    }
    for (const std::string& key : metadata_keys) {
        Ort::AllocatedStringPtr metadata_value = model_metadata.LookupCustomMetadataMapAllocated(key.c_str(), metadata_allocator);
        if (metadata_value != nullptr) {
            auto raw_metadata_value = metadata_value.get();
            metadata[key] = std::string(raw_metadata_value);
        }
    }
}

const std::vector<std::string>& OnnxModelBase::getInputNames() {
    return inputNodeNames;
}

const std::vector<std::string>& OnnxModelBase::getOutputNames() {
    return outputNodeNames;
}

const Ort::ModelMetadata& OnnxModelBase::getModelMetadata()
{
    return model_metadata;
}

const std::unordered_map<std::string, std::string>& OnnxModelBase::getMetadata()
{
    return metadata;
}

const Ort::Session& OnnxModelBase::getSession()
{
    return session;
}

const char* OnnxModelBase::getModelPath()
{
    return modelPath_;
}

const std::vector<const char*> OnnxModelBase::getOutputNamesCStr()
{
    return outputNamesCStr;
}

const std::vector<const char*> OnnxModelBase::getInputNamesCStr()
{
    return inputNamesCStr;
}

std::vector<Ort::Value> OnnxModelBase::forward(std::vector<Ort::Value>& inputTensors)
{
    return session.Run(Ort::RunOptions{ nullptr },
        inputNamesCStr.data(),
        inputTensors.data(),
        inputNamesCStr.size(),
        outputNamesCStr.data(),
        outputNamesCStr.size());
}
