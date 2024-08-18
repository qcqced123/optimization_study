from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import AutoQuantizationConfig


def do_dynamic_quantize(path: str):
    # do the dynamic quantization in ORT
    # get ORT format model from local hub
    # if you want to quantize the model (activation, weight) into quint8, use this optimum wrapping function
    quantizer = ORTQuantizer.from_pretrained(path)
    dq_config = AutoQuantizationConfig.avx512_vnni(
        is_static=False,
        per_channel=False
    )
    model_quantized_path = quantizer.quantize(
        save_dir="./saved/quantized/",
        quantization_config=dq_config,
    )
    return model_quantized_path
