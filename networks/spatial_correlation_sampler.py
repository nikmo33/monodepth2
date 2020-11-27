import torch
from torch import nn
from torch import autograd
from torch.nn.modules.utils import _pair


def spatial_correlation_sample(
    input1: torch.Tensor,
    input2: torch.Tensor,
    kernel_size: int = 1,
    patch_size: int = 1,
    stride: int = 1,
    padding: int = 0,
    dilation_patch: int = 1,
    keep_spatial_dims: bool = False,
    normalise: bool = True,
):
    """Apply spatial correlation sampling on from input1 to input2,

    Every parameter except input1 and input2 can be either single int
    or a pair of int. For more information about Spatial Correlation
    Sampling, see this page.
    https://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/

    Args:
        input1 : The first parameter.
        input2 : The second parameter.
        kernel_size : total size of your correlation kernel, in pixels
        patch_size : total size of your patch, determining how many
            different shifts will be applied
        stride : stride of the spatial sampler, will modify output
            height and width
        padding : padding applied to input1 and input2 before applying
            the correlation sampling, will modify output height and width
        dilation_patch : step for every shift in patch
        keep_spatial_dims : keep spatial dims to output (B x PatchH x PatchW x oH x oW)
        normalise : normalise by the channel input size of input 1

    Returns:
        Tensor: Result of correlation sampling

    """
    output = SpatialCorrelationSamplerFunction.apply(
        input1, input2, kernel_size, patch_size, stride, padding, dilation_patch
    )
    if not keep_spatial_dims:
        output_shape = output.shape
        output = output.view(output_shape[0], output_shape[1] * output_shape[2], output_shape[3], output_shape[4])
    if normalise:
        return output / input1.size(1)
    return output


class SpatialCorrelationSamplerFunction(autograd.Function):
    def __init__(self):
        super().__init__()
        load_cpp_library("libcppcorrelation.so")

    @staticmethod
    def forward(
        ctx,
        input1: torch.tensor,
        input2: torch.tensor,
        kernel_size: int,
        patch_size: int,
        stride: int,
        padding: int,
        dilation_patch: int,
    ):

        ctx.kernel_size = _pair(kernel_size)
        ctx.patch_size = _pair(patch_size)
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation_patch = _pair(dilation_patch)

        ctx.save_for_backward(input1, input2)
        kH, kW = ctx.kernel_size
        patchH, patchW = ctx.patch_size
        padH, padW = ctx.padding
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride

        output = torch.ops.wayve_ops.correlation_sample_forward(
            input1, input2, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW
        )
        return output

    @staticmethod
    @autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors  # pylint: disable=unpacking-non-sequence

        kH, kW = ctx.kernel_size
        patchH, patchW = ctx.patch_size
        padH, padW = ctx.padding
        dilation_patchH, dilation_patchW = ctx.dilation_patch
        dH, dW = ctx.stride

        grad_input1, grad_input2 = torch.ops.wayve_ops.correlation_sample_backward(
            input1, input2, grad_output, kH, kW, patchH, patchW, padH, padW, dilation_patchH, dilation_patchW, dH, dW
        )
        return grad_input1, grad_input2, None, None, None, None, None


class SpatialCorrelationSampler(nn.Module):
    def __init__(
        self,
        kernel_size: int = 1,
        patch_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        dilation_patch: int = 1,
        keep_spatial_dims: bool = False,
        normalise: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.dilation_patch = dilation_patch
        self.keep_spatial_dims = keep_spatial_dims
        self.normalise = normalise

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        return spatial_correlation_sample(
            input1,
            input2,
            self.kernel_size,
            self.patch_size,
            self.stride,
            self.padding,
            self.dilation_patch,
            self.keep_spatial_dims,
            self.normalise,
        )