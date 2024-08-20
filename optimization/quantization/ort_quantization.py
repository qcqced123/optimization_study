import os
import onnx
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer

from onnx import helper, numpy_helper, TensorProto
from onnxruntime.quantization import matmul_4bits_quantizer, matmul_bnb4_quantizer


def int8_quantize():
    """ weight, node, input-wise quantization, fp32 to int8 & uint8 manually
    because ONNX Quantization library does not support sub module-wise quantization
    """
    return


def bfloat16_quantize():
    """ weight, node, input-wise quantization, fp32 to int8 & uint8 manually
    because ONNX Quantization library does not support sub module-wise quantization

    """
    return


def rtn_4bit_quantize(
    model_name: str,
    block_size: int,
    accuracy_level: int,
    algo_config: matmul_4bits_quantizer.RTNWeightOnlyQuantConfig
) -> None:
    """ RTN quantization function, default target bit setting is 4bit
    Args:
        model_name (str):
        block_size (int):
        accuracy_level (int):
        algo_config:
    """
    model_output = f""
    ort_model = onnx.load()
    ort_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    ort_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
        model=ort_model,
        block_size=block_size,
        accuracy_level=accuracy_level,
        algo_config=algo_config
    )
    quant.process()
    ort_config.save_pretrained(model_output)
    ort_tokenizer.save_pretrained(model_output)
    quant.model.save_model_to_file(os.path.join(model_output, model_name), use_external_data_format=True)
    return


def qlora_4bit_quantize():
    return


if __name__ == '__main__':
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    algo_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()
    rtn_4bit_quantize(
        model_name=model_name,
        block_size=128,
        accuracy_level=4,
        algo_config=algo_config
    )
