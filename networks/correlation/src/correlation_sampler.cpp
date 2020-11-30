#include <vector>
#include <iostream>

#include "torch/script.h"

// declarations

at::Tensor correlation_cuda_forward(at::Tensor input1,
                                    at::Tensor input2,
                                    int kH,
                                    int kW,
                                    int patchH,
                                    int patchW,
                                    int padH,
                                    int padW,
                                    int dilation_patchH,
                                    int dilation_patchW,
                                    int dH,
                                    int dW);

at::Tensor correlation_cpp_forward(at::Tensor input1,
                                   at::Tensor input2,
                                   int kH,
                                   int kW,
                                   int patchH,
                                   int patchW,
                                   int padH,
                                   int padW,
                                   int dilation_patchH,
                                   int dilation_patchW,
                                   int dH,
                                   int dW);

std::vector<at::Tensor> correlation_cuda_backward(at::Tensor grad_output,
                                                  at::Tensor input1,
                                                  at::Tensor input2,
                                                  int kH,
                                                  int kW,
                                                  int patchH,
                                                  int patchW,
                                                  int padH,
                                                  int padW,
                                                  int dilation_patchH,
                                                  int dilation_patchW,
                                                  int dH,
                                                  int dW);

std::vector<at::Tensor> correlation_cpp_backward(at::Tensor grad_output,
                                                 at::Tensor input1,
                                                 at::Tensor input2,
                                                 int kH,
                                                 int kW,
                                                 int patchH,
                                                 int patchW,
                                                 int padH,
                                                 int padW,
                                                 int dilation_patchH,
                                                 int dilation_patchW,
                                                 int dH,
                                                 int dW);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

at::Tensor correlation_sample_forward(at::Tensor input1,
                                      at::Tensor input2,
                                      int64_t kH,
                                      int64_t kW,
                                      int64_t patchH,
                                      int64_t patchW,
                                      int64_t padH,
                                      int64_t padW,
                                      int64_t dilation_patchH,
                                      int64_t dilation_patchW,
                                      int64_t dH,
                                      int64_t dW) {
  if (input1.type().is_cuda()) {
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);

    return correlation_cuda_forward(
        input1, input2, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW);
  } else {
    return correlation_cpp_forward(
        input1, input2, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW);
  }
}

std::vector<at::Tensor> correlation_sample_backward(at::Tensor input1,
                                                    at::Tensor input2,
                                                    at::Tensor grad_output,
                                                    int64_t kH,
                                                    int64_t kW,
                                                    int64_t patchH,
                                                    int64_t patchW,
                                                    int64_t padH,
                                                    int64_t padW,
                                                    int64_t dilation_patchH,
                                                    int64_t dilation_patchW,
                                                    int64_t dH,
                                                    int64_t dW) {
  if (grad_output.type().is_cuda()) {
    CHECK_INPUT(input1);
    CHECK_INPUT(input2);
    return correlation_cuda_backward(
        input1, input2, grad_output, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW);
  } else {
    return correlation_cpp_backward(
        input1, input2, grad_output, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW);
  }
}

static auto registry = torch::RegisterOperators("wayve_ops::correlation_sample_forward", &correlation_sample_forward)
                           .op("wayve_ops::correlation_sample_backward", &correlation_sample_backward);
