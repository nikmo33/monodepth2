#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>

#include "torch/script.h"

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument("Error: You need to provide a model path as argument");
    }
    const char* model_path = argv[1];

    std::cout << "Loading traced model" << std::endl;
    torch::jit::script::Module module = torch::jit::load(model_path);
    std::cout << "Traced model loaded" << std::endl;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3, 10, 10}));
    inputs.push_back(torch::randn({1, 3, 10, 10}));

    std::cout << "Doing forward pass" << std::endl;
    torch::Tensor output = module.forward(std::move(inputs)).toTensor();
    std::cout << "Forward pass working" << std::endl;
  } catch (std::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  return 0;
}
