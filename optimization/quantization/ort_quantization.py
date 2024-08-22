import os
import onnx
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer
from onnx import helper, numpy_helper, TensorProto
from onnxruntime.quantization import matmul_4bits_quantizer, matmul_bnb4_quantizer


def int8_quantize(model_name: str, target_weight_name: str):
    """ weight, node, input-wise quantization, fp32 to int8 & uint8 manually
    because ONNX Quantization library does not support sub module-wise, node-wise quantization

    Args:
        model_name (str): path of local model hub
        target_weight_name (str):
    """
    # init scale, zero-point value for de-quantized node
    # load the ONNX format model
    scale = None
    zero_point = None
    onnx_model = onnx.load(model_name)

    # loop for find target_weight_name in onnx model initializer
    # and then quantize into 8bit
    for initializer in tqdm(onnx_model.graph.initializer):
        name = initializer.name
        data_type = initializer.data_type
        data_type_str = TensorProto.DataType.Name(data_type)
        print(f"ORT model weight name: {name}, Data Type: {data_type_str}", end="\n\n")

        if name == target_weight_name:
            weight_data = numpy_helper.to_array(initializer)

            min_val = np.min(weight_data)
            max_val = np.max(weight_data)

            scale = (max_val - min_val) / 255.0
            zero_point = int(-min_val / scale)

            weight_data_uint8 = ((weight_data / scale) + zero_point).astype(np.uint8)
            new_initializer = numpy_helper.from_array(weight_data_uint8, name=initializer.name)

            # replace original weight to new quantized weight
            onnx_model.graph.initializer.remove(initializer)
            onnx_model.graph.initializer.append(new_initializer)

    print(f"current scale factor is: {scale}")
    print(f"current zero point is: {zero_point}")

    # scale_value, zero_point_value will be calculated per
    scale_initializer = helper.make_tensor(
        name="embedding_scale",
        data_type=TensorProto.FLOAT,
        dims=[],
        vals=[scale]
    )
    zero_point_initializer = helper.make_tensor(
        name="embedding_zero_point",
        data_type=TensorProto.UINT8,
        dims=[],
        vals=[zero_point]
    )
    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["model.embed_tokens.weight_transposed", "embedding_scale", "embedding_zero_point"],
        outputs=["dequantized_weight"],
        name="Dequantize_Embed_Tokens_Weight"
    )

    # make quantized node
    # if current node is MatMul in lm_head(embed_tokens.weight_transposed), use the de-quantized weight of Matmul
    new_nodes = []
    for node in onnx_model.graph.node:
        if node.op_type == "MatMul" and "model.embed_tokens.weight_transposed" in node.input:
            new_inputs = ["dequantized_weight" if inp == "model.embed_tokens.weight_transposed" else inp for inp in node.input]
            new_node = helper.make_node(
                node.op_type,
                inputs=new_inputs,
                outputs=node.output,
                name=node.name
            )
            new_nodes.append(new_node)
        else:
            new_nodes.append(node)

    # remove original node (not quantized)
    # add new node (quantized)
    # add de-quantized for applying model to Q-DQ Style
    # add scale and zero-point value to Initializer
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)
    onnx_model.graph.node.append(dequantize_node)
    onnx_model.graph.initializer.extend([scale_initializer, zero_point_initializer])
    return


def fp32_to_bf16(fp32_array: np.ndarray) -> np.ndarray:
    """ helper function for converting fp32 model's weight into bf16
    Args:
        fp32_array (np.ndarray): numpy ndarray of target modules fp32 weight

    Reference:
        https://onnx.ai/onnx/
        https://onnxruntime.ai/docs/
    """
    bf16_array = np.right_shift(fp32_array.view(np.uint32), 16).astype(np.uint16)
    return bf16_array


def bfloat16_quantize(model_name: str, target_weight_name: str = "model.embed_tokens.weight") -> None:
    """ fp32 node-wise, initializer-wise(weight-wise) quantize to bf16 function

    bf16 is alias of bfloat16 or brain float16, made by Google 2019 for targeting Deep Learning
    ONNX support bf16, but currently some of operators such as "where" already contained several famous LLM in ONNX does not support bf16

    So, we cannot export the pytorch or tensorflow model into bf16 directly, left options are fp32, fp16
    ONNX fp16 does not support the CPU accelerator, when you use CPU accelerator, just only one option left (fp32)

    Many Quantization options or functions are already implemented in ONNX Library, but they do not quantize some module
    such as word embedding layer.

    So if you want to bf16 or other float precision of word embedding layer, you set up the precision to this layer manually
    this function will do that action exactly, and also have de-cast bf16 to fp32 node(operator) for weight-shared model
    (weight-shared & tied-embedding: word embedding's weight and lm head weight are the same and referencing each other)

    Args:
        model_name (str): path of local model hub
        target_weight_name (str): name of weight name to apply converting fp32 to bf16 for reducing model storage
                                  default value is model.embed_tokens.weight, indicating word embedding layers weight

    Reference:
        https://onnx.ai/onnx/
        https://onnxruntime.ai/docs/
    """
    # load onnx format model from local model hub
    # find the target node's weight to converting fp32 to bf16
    # convert fp32 weight to bf16
    onnx_model = onnx.load(model_name)
    for initializer in tqdm(onnx_model.graph.initializer):
        name = initializer.name
        data_type = initializer.data_type
        data_type_str = TensorProto.DataType.Name(data_type)

        if name == target_weight_name:
            print(f"ORT model weight name: {name}, Data Type: {data_type_str}", end="\n\n")

            # export target onnx module's weight into numpy ndarray
            # convert fp32 to bf16
            fp32_array = numpy_helper.to_array(initializer)
            bf16_array = fp32_to_bf16(fp32_array)

            # convert bf16 ndarray into onnx array
            bf16_initializer = numpy_helper.from_array(bf16_array, name=initializer.name)
            bf16_initializer.data_type = TensorProto.BFLOAT16

            # remove original(fp32) weight
            # add new(bf16) weight into graph
            onnx_model.graph.initializer.remove(initializer)
            onnx_model.graph.initializer.append(bf16_initializer)

    # initialize the name of target node
    # initialize the name of target node's weight name
    target_node_name = "Transpose_5369"
    target_tensor_name = target_weight_name

    # initialize the de-cast layer's output name
    cast_node_output_name = target_tensor_name + "_fp32"
    cast_node = onnx.helper.make_node(
        "Cast",
        inputs=[target_tensor_name],  # define the input: bf16 weight
        outputs=[cast_node_output_name],  # define the output: fp32 weight
        to=onnx.TensorProto.FLOAT,  # define the target dtype, (FLOAT = fp32)
        name="Cast_BF16_to_FP32"
    )

    # make the new graph node, including the de-cast node
    # add the de-cast node into new graph node
    # remove the old graph node
    # add the new graph node, including the de-cast node
    # save the new graph model in local model hub
    new_nodes = []
    for node in tqdm(onnx_model.graph.node):
        if node.name == target_node_name:
            new_inputs = [cast_node_output_name if inp == target_tensor_name else inp for inp in node.input]
            modified_node = onnx.helper.make_node(
                node.op_type,
                inputs=new_inputs,
                outputs=node.output,
                name=node.name
            )
            new_nodes.append(modified_node)
        else:
            new_nodes.append(node)

    new_nodes.insert(0, cast_node)
    onnx_model.graph.ClearField("node")
    onnx_model.graph.node.extend(new_nodes)

    bf16_model_path = "embed_bf16_w4_model.onnx"
    onnx.save(onnx_model, bf16_model_path)
    return


def rtn_4bit_quantize(
    model_name: str,
    block_size: int,
    accuracy_level: int,
    algo_config: matmul_4bits_quantizer.RTNWeightOnlyQuantConfig
) -> None:
    """ RTN quantization function, default target bit setting is 4bit
    This algorithm doesn't quantize the word embedding, language modeling head

    if you want to quantize those modules, you must do manually
    Args:
        model_name (str):
        block_size (int):
        accuracy_level (int):
        algo_config:
    """
    output_path = "./saved/onnx/4bit_precision/RTN/phi3"
    model_output = f"{output_path}/4bit_rtn_phi3.onnx"
    ort_model = onnx.load(model_name)
    ort_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    ort_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quant = matmul_4bits_quantizer.MatMul4BitsQuantizer(
        model=ort_model,
        block_size=block_size,
        accuracy_level=accuracy_level,
        algo_config=algo_config
    )
    quant.process()
    ort_config.save_pretrained(output_path)
    ort_tokenizer.save_pretrained(output_path)
    quant.model.save_model_to_file(os.path.join(model_output, model_name), use_external_data_format=True)
    return


def qlora_4bit_quantize(
    model_name: str,
    quant_type: str = "nf4",
    block_size: int = 64,
):
    """ QLoRA quantization function, default target bit setting is Normal Float 4bit
    This algorithm doesn't quantize the word embedding, language modeling head

    if you want to quantize those modules, you must do manually
    
    Args:
        model_name (str):
        quant_type (str):
        block_size (int):
    """
    output_path = "./saved/onnx/4bit_precision/RTN/phi3"
    model_output = f"{output_path}/4bit_rtn_phi3.onnx"

    ort_model = onnx.load(model_name)
    ort_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    ort_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # quant_type = 0 => FP4, quant_type = 1 => NF4
    # default block size is 64 in original paper
    quant = matmul_bnb4_quantizer.MatMulBnb4Quantizer(
        model=ort_model,
        quant_type=1 if quant_type == "nf4" else 0,
        block_size=block_size,
    )

    quant.process()
    ort_config.save_pretrained(output_path)
    ort_tokenizer.save_pretrained(output_path)
    quant.model.save_model_to_file(os.path.join(model_output, model_name), use_external_data_format=True)
    return


if __name__ == '__main__':
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    q_config = matmul_4bits_quantizer.RTNWeightOnlyQuantConfig()
    rtn_4bit_quantize(
        model_name=model_name,
        block_size=128,
        accuracy_level=4,
        algo_config=q_config
    )
