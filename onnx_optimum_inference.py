from typing import List, Dict, Tuple
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoConfig, AutoTokenizer, pipeline


def get_ort_config(path: str):
    return AutoConfig.from_pretrained(
        path,
        trust_remote_code=True
    )


def get_ort_tokenizer(path: str) -> AutoTokenizer:
    """ function for loading the saved tokenizer from local ONNX format path

    Args:
        path (str): model saved path in local hub

    """
    return AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
    )


def get_ort_model(path: str) -> ORTModelForCausalLM:
    """ function for returning the converted model into ONNX format, saved in local hub disk

    Args:
        path (str): model saved path in local hub
    """
    return ORTModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        use_cache=True,
        use_io_binding=True
    )


def get_pipe(ort_model: ORTModelForCausalLM, ort_config: AutoConfig, ort_tokenizer: AutoTokenizer) -> pipeline:
    return pipeline(
        task="text-generation-with-past",
        model=ort_model,
        config=ort_config,
        tokenizer=ort_tokenizer,
        device="cpu",
        # torch_dtype="torch.bfloat16",
        trust_remote_code=True,
    )


def do_inference(pipe: pipeline, inputs: str, sampling_params: Dict) -> str:
    return pipe(
        inputs,
        max_new_tokens=sampling_params["max_new_tokens"],
        do_sample=sampling_params["do_sample"],
        top_k=sampling_params["top_k"],
        top_p=sampling_params["top_p"],
    )


if __name__ == '__main__':
    model_path = "./saved/W4A16-onnx-stage5-eeve-phi3.5-mini-instruct"
    ort_config = get_ort_config(model_path)
    ort_tokenizer = get_ort_tokenizer(model_path)
    ort_model = get_ort_model(model_path)
    pipe = get_pipe(ort_model, ort_config, ort_tokenizer)
    sampling_params = {
        "max_new_tokens": 10,
        "do_sample": False,
        "top_k": 50,
        "top_p": 0.90,
    }

    prompts = "Hello, what is your name? How was your day?"
    outputs = do_inference(pipe, prompts, sampling_params)
    print(outputs)

