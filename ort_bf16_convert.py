""" python module for converting ONNX model weights and nodes to bfloat16, except Where nodes
Where node is not supported in ONNX with bfloat16
"""
import onnx
import numpy as np

from tqdm.auto import tqdm
from onnx import helper, TensorProto, ModelProto


def check_initializer(onnx_model: ModelProto) -> None:
    """ helper function for printing current input model's initializer(weight) name, dtype

    Args:
        onnx_model (ModelProto): input onnx model
    """
    for initializer in tqdm(onnx_model.graph.initializer):
        name = initializer.name
        data_type = initializer.data_type
        data_type_str = TensorProto.DataType.Name(data_type)
        print(f"ORT model weight name: {name}, Data Type: {data_type_str}", end="\n")


def check_node(onnx_model: ModelProto) -> None:
    """ helper function for printing current input model's node name

    Args:
        onnx_model (ModelProto): input onnx model
    """
    for node in tqdm(onnx_model.graph.node):
        name = node.name
        print(f"ORT model node name: {name}", end="\n")


def fp32_to_bf16(fp32_array: np.ndarray) -> np.ndarray:
    """ convert function for float32 numpy array to bfloat16 using bit-shifting

    Args:
        fp32_array (np.ndarray): input float32 numpy array
    """
    bf16_array = np.right_shift(fp32_array.view(np.uint32), 16).astype(np.uint16)
    return bf16_array


def convert_tensor_to_bfloat16(tensor) -> None:
    """ convert function for ONNX tensor from float32 to bfloat16

    Args:
        tensor (TensorProto): input ONNX tensor from ModelProto object (initializer)
    """

    # branch for float_data field
    if len(tensor.float_data) > 0:
        fp32_array = np.array(tensor.float_data, dtype=np.float32)
        bfloat16_data = fp32_to_bf16(fp32_array)

        tensor.ClearField('float_data')
        tensor.raw_data = bfloat16_data.tobytes()
        tensor.data_type = onnx.TensorProto.BFLOAT16

    # branch for raw_data field
    elif len(tensor.raw_data) > 0:
        fp32_array = np.frombuffer(tensor.raw_data, dtype=np.float32)
        bfloat16_data = fp32_to_bf16(fp32_array)

        tensor.raw_data = bfloat16_data.tobytes()
        tensor.data_type = onnx.TensorProto.BFLOAT16
    else:
        raise ValueError("The tensor has no float_data or raw_data field to convert.")


def convert_model_weights_to_bfloat16(onnx_model: ModelProto) -> ModelProto:
    """ convert function for weights in the ONNX model to bfloat16

    Args:
        onnx_model (ModelProto): input onnx model
    """
    for initializer in onnx_model.graph.initializer:
        if initializer.data_type == onnx.TensorProto.FLOAT:
            convert_tensor_to_bfloat16(initializer)

    return onnx_model


def convert_non_where_nodes_to_bfloat16(onnx_model: ModelProto) -> ModelProto:
    """ convert function for non-Where node outputs to bfloat16
    Where node is not supported in ONNX with bfloat16

    Args:
        onnx_model (ModelProto): input onnx model
    """
    for node in onnx_model.graph.node:
        if node.op_type != "Where":
            for idx in range(len(node.output)):
                output_value_info = helper.make_tensor_value_info(node.output[idx], onnx.TensorProto.BFLOAT16, None)
                onnx_model.graph.value_info.extend([output_value_info])

    return onnx_model


def convert_model_except_where_to_bfloat16(model_path: str, output_path: str) -> ModelProto:
    """ convert function for all weights and nodes except Where nodes to bfloat16

    Args:
        model_path (str): input onnx model path
        output_path (str): output onnx model path
    """
    onnx_model = onnx.load(model_path)
    onnx_model = convert_model_weights_to_bfloat16(onnx_model)
    onnx_model = convert_non_where_nodes_to_bfloat16(onnx_model)

    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_path + ".data",
        size_threshold=1024
    )
    print(f"Model saved to {output_path}")
    return onnx_model


if __name__ == "__main__":
    MODEL_PATH = "./saved/onnx-stage5-eeve-phi3.5-mini-instruct/model.onnx"
    OUTPUT_PATH = "model.onnx"
    onnx_model = convert_model_except_where_to_bfloat16(MODEL_PATH, OUTPUT_PATH)

    check_initializer(onnx_model)
    check_node(onnx_model)