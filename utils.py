import logging
import sys
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from trl import SFTConfig
from typing import List, Tuple, Optional
import datetime


class ColoredFormatter(logging.Formatter):
    # ANSI escape codes for colors
    blue = "\x1b[34;1m"  # 加粗蓝色
    green = "\x1b[32;1m"  # 加粗绿色
    yellow = "\x1b[33;1m"  # 加粗黄色
    red = "\x1b[31;1m"  # 加粗红色
    bold_red = "\x1b[31;1m"  # 更明显的加粗红色
    reset = "\x1b[0m"  # 重置颜色

    FORMATS = {
        logging.DEBUG: blue + "%(asctime)s - %(levelname)s" + reset + " - %(name)s - %(message)s",
        logging.INFO: green + "%(asctime)s - %(levelname)s" + reset + " - %(name)s - %(message)s",
        logging.WARNING: yellow + "%(asctime)s - %(levelname)s" + reset + " - %(name)s - %(message)s",
        logging.ERROR: red + "%(asctime)s - %(levelname)s" + reset + " - %(name)s - %(message)s",
        logging.CRITICAL: bold_red + "%(asctime)s - %(levelname)s" + reset + " - %(name)s - %(message)s",
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(
            record.levelno, self.FORMATS[logging.INFO]
        )  # Default to INFO format if not specified
        formatter = logging.Formatter(log_fmt, "%m/%d/%Y %H:%M:%S")
        return formatter.format(record)


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter())
    return console_handler


def get_file_handler(logging_file):
    standard_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", "%m/%d/%Y %H:%M:%S")
    file_handler = logging.FileHandler(logging_file)
    file_handler.setFormatter(standard_formatter)
    return file_handler


@dataclass
class CustomConfig:
    # base model
    pretrained_model_name_or_path: str

    # quantization: 不兼容HfParser
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"

    # peft: 很多 data type定义不兼容HfParser
    task_type: str = "CAUSAL_LM"
    lora_r: Optional[int] = field(default=8)
    target_modules: List[str] = field(default_factory=list)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)

    # generate
    # dataset
    dataset: str = ""


def parse_config(config_file: str) -> Tuple[CustomConfig, SFTConfig]:
    parser = HfArgumentParser([CustomConfig, SFTConfig])
    custom_config, sft_config = parser.parse_yaml_file(config_file)

    sft_config.output_dir = (
        custom_config.pretrained_model_name_or_path.split("/")[-1]
        + "/"
        + custom_config.dataset.split("/")[-1]
        + "/"
        + datetime.datetime.now().strftime("%y__%m_%d__%H_%M")
    )
    sft_config.__post_init__()
    return custom_config, sft_config


if __name__ == "__main__":
    custom_config, sft_config = parse_config("config/sft.yaml")
    print(custom_config)
    print(sft_config)
