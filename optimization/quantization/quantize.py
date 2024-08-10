import torch
from torch import Tensor


def absmax_quantize(x: Tensor) -> Tensor:
    """ Absmax: Absolute Maximum Quantization (FP32 to INT8)
    This method is useful when the input has a symmetric distribution

    Algorithm:
        1) divide X(original tensor) by absolute maximum one in X
        2) multiply the INT8 scaling factor 127
        3) round the result of step 2 for converting INT8
        4) cast dtype to torch.int8
    """
    absmax = torch.max(torch.abs(x))
    x_q = (127 / absmax * x).round()
    return x_q.to(torch.int8)


def zero_point_quantize(x: Tensor) -> Tensor:
    """ Zero-point Quantization
    This method is useful when the input has a asymmetric distibutions (e.g passed result from ReLU, GeLU ...)

    Algorithm:
        1) calculate the scale factor:
            - range of target bit (e.g: INT8 -> 255)
            - range / max(X) - min(X)
        2) get the zero-point: -round(scale*min(X)) - 128
        3) quantize the original value: round(scale * X + zero-point)
    """
    maximum, minimum = torch.max(x), torch.min(x)
    target_range = maximum - minimum if maximum - minimum else 1  # for numerical stability

    scale = 255 / target_range
    zero_point = (-scale * minimum - 128).round()
    x_q = torch.clamp(
        input=(scale * x + zero_point).round(),
        min=-128,
        max=127
    )
    return x_q.to(torch.int8)

