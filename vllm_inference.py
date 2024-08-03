""" Initialize vLLM’s engine for offline inference with the LLM class
LLM class is the child of Huggingface AutoModel, so we can init this class same as Huggingface AutoModel usage
"""
import pandas as pd

from typing import List
from vllm import LLM, SamplingParams


def get_inputs(df: pd.DataFrame) -> List[str]:
    return [str(doc) for doc in df["doc"].tolist() if doc is not None]


def initialize_llm(model_name: str, max_length: int, q_method: str) -> LLM:
    return LLM(
        model=model_name,
        max_model_len=max_length,
        max_seq_len_to_capture=4096,
        quantization=q_method,
        trust_remote_code=True,
        swap_space=2
    )


def do_inference(llm: LLM, inputs: List[str], sampling_params: SamplingParams):
    return llm.generate(
        inputs,
        sampling_params=sampling_params
    )


if __name__ == '__main__':
    prompts = get_inputs(
        pd.read_csv("./optimization/dataset/2112.07076.csv")
    )

    model_name = "./awq/phi3"
    quantization_method = "AWQ"

    llm_model = initialize_llm(
        model_name=model_name,
        max_length=21153,
        q_method=quantization_method
    )

    sampling_params = SamplingParams(
        seed=42,
        max_tokens=32,
        temperature=0.5,
        top_k=50,
        top_p=0.90
    )
    outputs = do_inference(
        llm=llm_model,
        inputs=prompts,
        sampling_params=sampling_params
    )

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
