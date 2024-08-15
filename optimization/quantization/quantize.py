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


def quantile_quantize(x: Tensor, k: int) -> Tensor:
    """ function for implemented quantile quantization

    Args:
        x (torch.Tensor): input tensor for quantization
        k (int): value of target bit (k-bit quantization)

    Return:
        quantized tensor

    Example:
        x = torch.randn(32)
        y = quantile_quantize(x, 4)

        print(f"original tensor value is:", end='\n')
        print(f"{x}", end='\n\n')

        print(f"quantile quantized tensor value is:", end='\n')
        print(f"{y}", end='\n\n')

        original tensor value is:
        tensor([ 2.2941,  1.0962, -2.4873,  0.1702,  0.3427, -0.4277, -0.1280, -1.4415,
                 0.1170, -1.2307, -3.4868, -0.2875, -0.0901, -0.2967, -1.6805,  1.2236,
                 0.6677, -0.4470,  0.6157, -1.3107,  2.1138, -1.2380,  0.9999,  1.0647,
                 1.4255, -0.5064,  1.4726, -1.2967, -0.1502, -0.0469, -0.4975, -0.7803])

        quantile quantized tensor value is:
        tensor([ 1.5127,  1.2489, -1.7309,  0.4280,  0.4280, -0.2915, -0.0712, -1.3271,
                 0.1370, -0.8929, -1.7309, -0.1391, -0.0712, -0.2915, -1.3271,  1.2489,
                 0.7508, -0.4349,  0.7508, -1.2490,  1.5127, -0.8929,  1.0706,  1.0706,
                 1.5127, -0.5003,  1.5127, -1.2490, -0.1391,  0.1370, -0.4349, -0.5003])
    """
    nums = 2 ** k
    bins = torch.tensor([i / nums for i in range(1, nums)])
    quantiles = torch.quantile(x, q=bins)
    indices = torch.clamp(
        input=torch.bucketize(x, quantiles),
        max=quantiles.size(0) - 1
    )
    return quantiles[indices]

