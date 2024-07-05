from dataclasses import asdict, dataclass, field, fields
from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from typing import List


@dataclass
class CustomConfig:
    load_in_4bit: bool = False

    # lora
    r: int = 8
    target_modules: List[str] = field(default_factory=list)
    task_type: str = "CAUSAL_LM"


parser = HfArgumentParser([CustomConfig, TrainingArguments])  # loraconfig 有data type定义不兼容HfParser
custom_config, training_args = parser.parse_yaml_file("config.yaml")
print(custom_config)
