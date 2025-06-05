#include <torch/script.h>
#include <vector>
#include <Eigen/Dense>

class PolicyInference {
public:
    PolicyInference(const std::string& model_path,
                   const std::vector<float>& default_angles,
                   float action_scale)
        : default_angles_(default_angles),
          action_scale_(action_scale) {
        try {
            // Load the TorchScript model
            policy_ = torch::jit::load(model_path);
            policy_.eval();  // Set the model to evaluation mode
        }
        catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
        }
    }

    std::vector<float> infer(const std::vector<float>& obs) {
        // Convert observation to tensor
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor obs_tensor = torch::from_blob(
            const_cast<float*>(obs.data()),
            {1, static_cast<long>(obs.size())},
            options
        );

        // Perform inference
        torch::NoGradGuard no_grad;
        torch::Tensor action_tensor = policy_.forward({obs_tensor}).toTensor();

        // Convert tensor to vector
        std::vector<float> action(action_tensor.size(1));
        std::memcpy(action.data(),
                   action_tensor.data_ptr<float>(),
                   action_tensor.size(1) * sizeof(float));

        // Transform action to target_dof_pos
        std::vector<float> target_dof_pos(action.size());
        for (size_t i = 0; i < action.size(); ++i) {
            target_dof_pos[i] = action[i] * action_scale_ + default_angles_[i];
        }

        return target_dof_pos;
    }

private:
    torch::jit::script::Module policy_;
    std::vector<float> default_angles_;
    float action_scale_;
};

// Example usage
int main() {
    // Initialize parameters
    std::string model_path = "path/to/your/model.pt";
    std::vector<float> default_angles = {0.0f, 0.0f, 0.0f}; // Example values
    float action_scale = 1.0f;

    // Create policy inference object
    PolicyInference policy(model_path, default_angles, action_scale);

    // Example observation vector
    std::vector<float> obs = {0.0f, 0.0f, 0.0f}; // Example values

    // Perform inference
    std::vector<float> target_dof_pos = policy.infer(obs);

    return 0;
}
