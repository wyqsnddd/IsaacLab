#include <NvInfer.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <fstream>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO) {
            std::cout << msg << std::endl;
        }
    }
} gLogger;

class PolicyInference {
public:
    PolicyInference(const std::string& engine_path,
                   const std::vector<float>& default_angles,
                   float action_scale)
        : default_angles_(default_angles),
          action_scale_(action_scale) {
        // Load TensorRT engine
        std::ifstream file(engine_path, std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Failed to open engine file");
        }

        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(size);
        file.read(engineData.data(), size);
        file.close();

        runtime_ = nvinfer1::createInferRuntime(gLogger);
        engine_ = runtime_->deserializeCudaEngine(engineData.data(), size);
        context_ = engine_->createExecutionContext();

        // Allocate GPU memory
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
    }

    ~PolicyInference() {
        cudaFree(input_buffer_);
        cudaFree(output_buffer_);
        context_->destroy();
        engine_->destroy();
        runtime_->destroy();
    }

    std::vector<float> infer(const std::vector<float>& obs) {
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
            target_dof_pos[i] = action[i] * action_scale_ + default_angles_[i];
        }

        return target_dof_pos;
    }

private:
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    void* input_buffer_;
    void* output_buffer_;
    nvinfer1::Dims input_dims_;
    nvinfer1::Dims output_dims_;
    size_t input_size_;
    size_t output_size_;
    std::vector<float> default_angles_;
    float action_scale_;
};

// Example usage
int main() {
    // Initialize parameters
    std::string engine_path = "path/to/your/model.engine";
    std::vector<float> default_angles = {0.0f, 0.0f, 0.0f}; // Example values
    float action_scale = 1.0f;

    try {
        // Create policy inference object
        PolicyInference policy(engine_path, default_angles, action_scale);

        // Example observation vector
        std::vector<float> obs = {0.0f, 0.0f, 0.0f}; // Example values

        // Perform inference
        std::vector<float> target_dof_pos = policy.infer(obs);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
