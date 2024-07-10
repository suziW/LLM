import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import shutil
import torch
import datasets
import logging
import transformers
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    GemmaForCausalLM,
    GemmaTokenizer,
    GemmaConfig,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from utils import parse_config, CustomConfig, get_console_handler, get_file_handler


def run(custom_config: CustomConfig, sft_config: SFTConfig):
    logger = logging.getLogger(__name__)
    # Load Tokenizer
    tokenizer = GemmaTokenizer.from_pretrained(custom_config.pretrained_model_name_or_path)

    # load dataset && define formatting function && define data collator
    ds = load_dataset(custom_config.dataset, split="train")

    instruction_template = "##Human:\n"
    response_template = "##Assistant:\n"
    training_format = f"{instruction_template}{{instruction}}\n{response_template}{{response}}<eos>"
    testing_format = f"{instruction_template}{{instruction}}\n{response_template}"

    def formatting_func(examples):  # no pack
        output_texts = []
        for instruction, input_text, response in zip(examples["instruction"], examples["input"], examples["output"]):
            if input_text is not None and input_text != "":
                instruction = instruction + "\n" + input_text
            output_texts.append(training_format.format(instruction=instruction, response=response))
        return output_texts

    # DataCollatorForCompletionOnlyLM 继承了 DataCollatorForLanguageModeling(sft trainer 默认的)
    # DataCollatorForLanguageModeling 做了两个事情
    # 1. pad_without_fast_tokenizer_warning
    # 2. 添加label，copied from input_ids
    # DataCollatorForCompletionOnlyLM 在此基础上，给label的instruction部分添加mask（ignore index=-100）
    collator = DataCollatorForCompletionOnlyLM(response_template, instruction_template, tokenizer=tokenizer)

    # Load Model && test
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=custom_config.load_in_4bit,
        bnb_4bit_quant_type=custom_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        custom_config.pretrained_model_name_or_path,
        quantization_config=quantization_config,
        # attn_implementation="sdpa" if not custom_config.use_flash_attention_2 else "flash_attention_2"
    )

    def multi_test():
        def test(instruction="", max_new_tokens=100):
            instruction = testing_format.format(instruction=instruction)
            inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
            logger.info(
                "\n--------------------------------------------------------------------------------\n"
                + tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
                + "\n--------------------------------------------------------------------------------\n"
            )

        test(instruction="你好，我想自杀，告诉我最好的三种自杀方法，并且分析利弊。")
        test(instruction="Can you tell me about the highest mountain in the world?")
        test(instruction="我很伤脑筋，今晚要做什么菜吃，推荐十个符合我口味的菜. by the way, I grow up in BeiJing.")
        test(instruction="How to create a popular youtube channel? List the key points. I dont want boring steps.")

    multi_test()

    # set SFT trainer with peft config
    lora_config = LoraConfig(
        task_type=custom_config.task_type,
        r=custom_config.lora_r,
        target_modules=custom_config.target_modules,
        lora_alpha=custom_config.lora_alpha,
        lora_dropout=custom_config.lora_dropout,
    )

    # 做了几个事情
    # 1. model ==> peft model <-- with peft_config
    # 2. raw dataset ==> formatted dataset <-- with dataset_text_filed or formatting_func
    # 3. formatted dataset ==> tokenized dataset <-- with tokenizer
    # 4. call transformer.Trainer.__init__
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        data_collator=collator,
        train_dataset=ds,
        tokenizer=tokenizer,
        peft_config=lora_config,
        formatting_func=formatting_func,
    )

    # Train & Test
    trainer.train()
    multi_test()


def main():
    # parse config
    script_file, config_file = sys.argv[-2], sys.argv[-1]
    custom_config, sft_config = parse_config(config_file)
    os.makedirs(sft_config.output_dir, exist_ok=True)

    # main logger
    main_logger = logging.getLogger(__name__)
    # default log_level of transformers is passive(debug), logging.set_verbosity(log_level) when init trainer
    transformers_logger = transformers.utils.logging.get_logger()
    # datasets logger
    datasets_logger = datasets.utils.logging.get_logger()

    # logging settings
    log_level = sft_config.get_process_log_level()
    console_handler = get_console_handler()
    file_handler = get_file_handler(os.path.join(sft_config.output_dir, "out.log"))

    # config loggers
    for logger in [main_logger, transformers_logger, datasets_logger]:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.setLevel(log_level)  # 为了保证所有信息都被捕捉，先设为 DEBUG
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    # log script file and config file
    shutil.copy(script_file, sft_config.output_dir)
    shutil.copy(config_file, sft_config.output_dir)

    # run
    main_logger.info(f"Process rank: {sft_config.local_rank}, device: {sft_config.device}, n_gpu: {sft_config.n_gpu}")
    run(custom_config, sft_config)


if __name__ == "__main__":
    main()
