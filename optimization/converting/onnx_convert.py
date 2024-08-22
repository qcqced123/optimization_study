from optimum.exporters.onnx import main_export
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def get_config(name: str) -> AutoConfig:
    return AutoConfig.from_pretrained(
        pretrained_model_name_or_path=name,
        trust_remote_code=True
    )


def get_tokenizer(name: str) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=name,
        trust_remote_code=True
    )


def get_model(name: str, config: AutoConfig, bit_config: BitsAndBytesConfig = None) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=name,
        config=config,
        quantization_config=bit_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
        torch_dtype="auto"
    )


def convert2onnx(name: str, config: AutoConfig) -> None:
    """ function to export the pytorch model into ONNX Format with optimum

    Args:
        name (str): path for local hub or model name in huggingface remote model hub
        config (AutoConfig): pretrain or fine-tune config file for exporting model
    """
    main_export(
        model_name_or_path=name,
        config=config,
        output="./saved/",
        task="text-generation-with-past",
        # opset=21,
        device="cpu",
        dtype="fp32",
        # optimize="O3",
        framework="pt",
        trust_remote_code=True,
    )


if __name__ == '__main__':
    model_name = "microsoft/Phi-3-mini-128k-instruct"
    config = get_config(model_name)
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name, config)
    convert2onnx(model_name, config)
