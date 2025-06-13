#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <yaml-cpp/yaml.h>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

// 配置结构体
struct NetworkConfig {
    int input_dim;
    int output_dim;
    std::vector<int> hidden_dims;
    std::string activation;
    float action_scale;
    std::vector<float> default_angles;
    size_t workspace_size;
    bool fp16;
    int max_batch_size;
    std::string onnx_path;
    std::string engine_path;
};

// 配置加载器
class ConfigLoader {
public:
    static NetworkConfig loadConfig(const std::string& config_path) {
        NetworkConfig config;
        try {
            YAML::Node yaml_config = YAML::LoadFile(config_path);

            // 加载网络配置
            config.input_dim = yaml_config["network"]["input_dim"].as<int>();
            config.output_dim = yaml_config["network"]["output_dim"].as<int>();
            config.hidden_dims = yaml_config["network"]["hidden_dims"].as<std::vector<int>>();
            config.activation = yaml_config["network"]["activation"].as<std::string>();

            // 加载动作配置
            config.action_scale = yaml_config["action"]["scale"].as<float>();
            config.default_angles = yaml_config["action"]["default_angles"].as<std::vector<float>>();

            // 加载 TensorRT 配置
            config.workspace_size = yaml_config["tensorrt"]["workspace_size"].as<size_t>();
            config.fp16 = yaml_config["tensorrt"]["fp16"].as<bool>();
            config.max_batch_size = yaml_config["tensorrt"]["max_batch_size"].as<int>();

            // 加载文件路径
            config.onnx_path = yaml_config["paths"]["onnx_model"].as<std::string>();
            config.engine_path = yaml_config["paths"]["engine_file"].as<std::string>();

        } catch (const YAML::Exception& e) {
            throw std::runtime_error("Failed to load config: " + std::string(e.what()));
        }
        return config;
    }
};

// TensorRT 引擎构建器类
class TensorRTBuilder {
public:
    TensorRTBuilder(const NetworkConfig& config)
        : config_(config) {
        builder_ = nvinfer1::createInferBuilder(gLogger);
        network_ = builder_->createNetworkV2(1U << static_cast<int>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH));
        config_ = builder_->createBuilderConfig();

        // 设置最大工作空间大小
        config_->setMaxWorkspaceSize(config_.workspace_size);

        // 启用 FP16 精度（如果支持）
        if (config_.fp16 && builder_->platformHasFastFp16()) {
            config_->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }

    ~TensorRTBuilder() {
        if (config_) config_->destroy();
        if (network_) network_->destroy();
        if (builder_) builder_->destroy();
    }

    bool build() {
        // 检查 engine 文件是否已存在且有效
        if (isEngineValid()) {
            std::cout << "Using existing TensorRT engine: " << config_.engine_path << std::endl;
            return true;
        }

        std::cout << "Building new TensorRT engine..." << std::endl;
        // 创建 ONNX 解析器
        auto parser = nvinfer1::createONNXParser(*network_, gLogger);
        if (!parser) {
            std::cerr << "Failed to create ONNX parser" << std::endl;
            return false;
        }

        // 解析 ONNX 文件
        if (!parser->parseFromFile(config_.onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::cerr << "Failed to parse ONNX file" << std::endl;
            parser->destroy();
            return false;
        }

        // 构建引擎
        auto engine = builder_->buildEngineWithConfig(*network_, *config_);
        if (!engine) {
            std::cerr << "Failed to build TensorRT engine" << std::endl;
            parser->destroy();
            return false;
        }

        // 序列化引擎
        auto serialized_engine = engine->serialize();
        if (!serialized_engine) {
            std::cerr << "Failed to serialize engine" << std::endl;
            engine->destroy();
            parser->destroy();
            return false;
        }

        // 保存引擎到文件
        std::ofstream engine_file(config_.engine_path, std::ios::binary);
        if (!engine_file) {
            std::cerr << "Failed to open engine file for writing" << std::endl;
            serialized_engine->destroy();
            engine->destroy();
            parser->destroy();
            return false;
        }

        engine_file.write(static_cast<const char*>(serialized_engine->data()), serialized_engine->size());
        engine_file.close();

        // 清理资源
        serialized_engine->destroy();
        engine->destroy();
        parser->destroy();

        return true;
    }

private:
    bool isEngineValid() {
        std::ifstream file(config_.engine_path, std::ios::binary);
        if (!file.good()) {
            return false;
        }

        // 读取文件头信息
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size == 0) {
            return false;
        }

        // 尝试加载引擎以验证其有效性
        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();

        auto runtime = nvinfer1::createInferRuntime(gLogger);
        if (!runtime) {
            return false;
        }

        auto engine = runtime->deserializeCudaEngine(engineData.data(), size);
        if (!engine) {
            runtime->destroy();
            return false;
        }

        // 清理资源
        engine->destroy();
        runtime->destroy();
        return true;
    }
    NetworkConfig config_;
    nvinfer1::IBuilder* builder_;
    nvinfer1::INetworkDefinition* network_;
    nvinfer1::IBuilderConfig* config_;
};

class PolicyInference {
public:
    PolicyInference(const NetworkConfig& config)
        : config_(config) {
        // 加载 TensorRT engine
        if (!loadEngine(config_.engine_path)) {
            throw std::runtime_error("Failed to load TensorRT engine");
        }

        // 验证输入输出维度
        if (input_dims_.d[1] != config_.input_dim) {
            throw std::runtime_error("Input dimension mismatch. Expected " +
                                   std::to_string(config_.input_dim) + ", got " +
                                   std::to_string(input_dims_.d[1]));
        }
        if (output_dims_.d[1] != config_.output_dim) {
            throw std::runtime_error("Output dimension mismatch. Expected " +
                                   std::to_string(config_.output_dim) + ", got " +
                                   std::to_string(output_dims_.d[1]));
        }
    }

    ~PolicyInference() {
        cudaFree(input_buffer_);
        cudaFree(output_buffer_);
        context_->destroy();
        engine_->destroy();
        runtime_->destroy();
    }

    std::vector<float> infer(const std::vector<float>& obs) {
        // 验证输入维度
        if (obs.size() != config_.input_dim) {
            throw std::runtime_error("Input size mismatch. Expected " +
                                   std::to_string(config_.input_dim) + ", got " +
                                   std::to_string(obs.size()));
        }

        // Copy input to GPU
        cudaMemcpy(input_buffer_, obs.data(), input_size_ * sizeof(float), cudaMemcpyHostToDevice);

        // Run inference
        void* bindings[] = {input_buffer_, output_buffer_};
        context_->executeV2(bindings);

        // Copy output from GPU
        std::vector<float> action(output_size_);
        cudaMemcpy(action.data(), output_buffer_, output_size_ * sizeof(float), cudaMemcpyDeviceToHost);

        // Transform action to target_dof_pos
        std::vector<float> target_dof_pos(action.size());
        for (size_t i = 0; i < action.size(); ++i) {
            target_dof_pos[i] = action[i] * config_.action_scale + config_.default_angles[i];
        }

        return target_dof_pos;
    }

private:
    bool loadEngine(const std::string& engine_path) {
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        if (!runtime_) {
            return false;
        }

        engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
        if (!engine_) {
            runtime_->destroy();
            return false;
        }

        context_ = engine_->createExecutionContext();
        if (!context_) {
            engine_->destroy();
            runtime_->destroy();
            return false;
        }

        // 分配 GPU 内存
        input_dims_ = engine_->getBindingDimensions(0);
        output_dims_ = engine_->getBindingDimensions(1);

        input_size_ = 1;
        for (int i = 0; i < input_dims_.nbDims; i++) {
            input_size_ *= input_dims_.d[i];
        }

        output_size_ = 1;
        for (int i = 0; i < output_dims_.nbDims; i++) {
            output_size_ *= output_dims_.d[i];
        }

        cudaMalloc(&input_buffer_, input_size_ * sizeof(float));
        cudaMalloc(&output_buffer_, output_size_ * sizeof(float));

        return true;
    }
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* input_buffer_;
    void* output_buffer_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    size_t input_size_;
    size_t output_size_;
    NetworkConfig config_;
};

// Example usage
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    try {
        // 加载配置
        NetworkConfig config = ConfigLoader::loadConfig(argv[1]);

        // 1. 构建或加载 TensorRT 引擎
        TensorRTBuilder builder(config);
        if (!builder.build()) {
            std::cerr << "Failed to build/load TensorRT engine" << std::endl;
            return 1;
        }

        // 2. 使用构建好的引擎进行推理
        PolicyInference policy(config);

        // Example observation vector
        std::vector<float> obs(config.input_dim, 0.0f);

        // Perform inference
        std::vector<float> target_dof_pos = policy.infer(obs);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
