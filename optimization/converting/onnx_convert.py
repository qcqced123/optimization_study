import torch
import torch.nn as nn

from optimum.exporters.onnx import main_export
from transformers import pipeline, BitsAndBytesConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


""" load the pretrained config, tokenizer, model from huggingface hub """

model_name = "microsoft/Phi-3-mini-128k-instruct"
config = AutoConfig.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True
)
bit_config = None
tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    config=config,
    quantization_config=bit_config,
    # attn_implementation="flash_attention_2",
    trust_remote_code=True,
    torch_dtype="auto"
)

""" convert pytorch model to onnx """

main_export(
    model_name_or_path=model_name,
    config=config,
    output="./saved/",
    task="text-generation-with-past",
    #opset=20,
    device="cpu",
    dtype="bf16",
    optimize="O3",
    framework="pt",
    trust_remote_code=True,
)

