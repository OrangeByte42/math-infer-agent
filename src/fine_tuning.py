import os
import re
import json
import time
from typing import Any, Dict, List

import torch
from datasets import DatasetDict, Dataset, load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, TaskType, PeftModel
from peft import get_peft_model

from src.configs.configs import GeneralConfigs, LlamaConfigs
from src.utils import show_devices_info, calculate_duration_time


def fine_tune_model(configs: GeneralConfigs, save_dir: str,
                    lora_configs: Dict[str, Any], training_configs: Dict[str, Any]) -> None:
    """ Fine-tune a pre-trained language model on math reasoning tasks."""

    start_ts: float = time.time()

    # Show device info
    print(f"=== Device Info ===")
    show_devices_info()

    # Set CUDNN Benchmark
    print(f"=== Set CUDNN Benchmark ===")
    torch.backends.cudnn.benchmark = True

    # Load Model & Tokenizer
    print(f"=== Load Model & Tokenizer ===")
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        configs.MODEL_ID,
        torch_dtype=configs.TORCH_DTYPE,
        device_map=configs.DEVICE_MAP,
    )

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(configs.MODEL_ID)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset
    print(f"=== Load Datasets ===")
    dataset_dict: DatasetDict = load_dataset(configs.DATASET_DICT_ID, name=configs.DATASET_NAME)
    training_dataset: Dataset = dataset_dict["train"]
    test_dataset: Dataset = dataset_dict["test"]

    def preprocess_function(example: Any) -> Dict[str, Any]:
        """ Preprocess the dataset examples by tokenizing and formatting them for training."""

        # Prepare the prompt and target
        prompt: str = configs.user_prompt(example['question']) + configs.assistant_todo_prompt()
        target: str = re.sub(r"<<.*?>>", "", example['answer']).strip() + configs.EOT_TOKEN
        full_text: str = prompt + target

        # Tokenize the prompt and target
        tokenized_full_text: Dict[str, Any] = tokenizer(
            full_text,
            max_length=configs.INPUT_MAX_LENGTH + configs.MAX_NEW_TOKENS,
            padding=False,
            truncation=True,
        )
        tokenized_prompt: Dict[str, Any] = tokenizer(
            prompt,
            max_length=configs.INPUT_MAX_LENGTH,
            padding=False,
            truncation=True,
        )

        # Create labels, masking the prompt part with -100
        prompt_length: int = len(tokenized_prompt['input_ids'])
        labels: List[int] = [-100] * prompt_length + tokenized_full_text['input_ids'][prompt_length:]

        return {
            'input_ids': tokenized_full_text['input_ids'],
            'attention_mask': tokenized_full_text['attention_mask'],
            'labels': labels,
        }

    # Preprocess the datasets
    print(f"=== Preprocess the Datasets ===")
    training_dataset: Dataset = training_dataset.map(preprocess_function, remove_columns=training_dataset.column_names)
    test_dataset: Dataset = test_dataset.map(preprocess_function, remove_columns=test_dataset.column_names)

    # Data Collator
    print(f"=== Data Collator ===")
    data_collator: DataCollatorForSeq2Seq = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )

    # Configure LoRA
    print(f"=== Configure LoRA ===")
    lora_config: LoraConfig = LoraConfig(**lora_configs)

    original_model_params: int = sum(p.numel() for p in model.parameters())
    model: PeftModel = get_peft_model(model, lora_config)
    model.train()
    peft_model_params: int = sum(p.numel() for p in model.parameters())
    trainable_lora_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total original model parameters:    {original_model_params:,}")
    print(f"Total PEFT model parameters:        {peft_model_params:,}")
    print(f"Total trainable LoRA parameters:    {trainable_lora_params:,}")
    print(f"\tPercentage in original model: {(100 * trainable_lora_params / original_model_params):.4f}% ({trainable_lora_params:,} / {original_model_params:,})")
    print(f"\tPercentage in PEFT model:     {(100 * trainable_lora_params / peft_model_params):.4f}% ({trainable_lora_params:,} / {peft_model_params:,})")

    # Configure Training Arguments
    print(f"=== Configure Training Arguments ===")
    # model.config.use_cache = False  # Disable cache for training
    training_args: TrainingArguments = TrainingArguments(**training_configs)

    # Initialize Trainer
    print(f"=== Initialize Trainer ===")
    trainer: Trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    # Start Training
    print(f"=== Start Training ===")
    trainer.train()

    print(f"=== Save the Fine-tuned LoRA Model ===")
    print(f"Saving model ...")
    model.save_pretrained(os.path.join(save_dir, f"lora-model"))
    tokenizer.save_pretrained(os.path.join(save_dir, f"lora-tokenizer"))
    print(f"Model saved.")

    end_ts: float = time.time()
    minutes, seconds = calculate_duration_time(start_ts, end_ts)

    print(f"=== Save Configurations ===")
    runtime_configs: Dict[str, Any] = {
        "Running Time": {
            "Start Timestamp (s)": start_ts,
            "End Timestamp (s)": end_ts,
            "Duration (mm:ss)": f"{minutes}m:{seconds}s ({int(end_ts - start_ts):,} s)",
        },
        "Training Statistics": {
            "Original Model Parameters": f"{original_model_params:,}",
            "PEFT Model Parameters": f"{peft_model_params:,}",
            "Trainable LoRA Parameters": f"{trainable_lora_params:,}",
            "Ratio in Original Model (%)": f"{(100 * trainable_lora_params / original_model_params):.4f}%",
            "Ratio in PEFT Model (%)": f"{(100 * trainable_lora_params / peft_model_params):.4f}%",
        },
        "Model Configurations": {
            "Model ID": configs.MODEL_ID,
            "Dataset Dict ID": configs.DATASET_DICT_ID,
            "Dataset Name": configs.DATASET_NAME,
            "Dataset Train Size": len(training_dataset),
            "Dataset Test Size": len(test_dataset),
            "Input Max Length": configs.INPUT_MAX_LENGTH,
            "Max New Tokens": configs.MAX_NEW_TOKENS,
        },
        "LoRA Configurations": lora_configs,
        "Training Configurations": training_configs,
    }

    configs_save_path: str = os.path.join(save_dir, f"lora-configs.jsonl")
    with open(configs_save_path, "w", encoding="utf-8") as f:
        json.dump(runtime_configs, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    # Avoid potential model download issues
    os.environ['HF_HUB_OFFLINE'] = '1'

    # For Llama-3.2-1B-Instruct
    configs: GeneralConfigs = LlamaConfigs(
        MODEL_ID="meta-llama/Llama-3.2-1B-Instruct",
        DEVICE_MAP="auto",
    )

    save_dir: str = os.path.join(".", f"LoRA-{configs.MODEL_NAME}-{configs.DATASET_DICT_NAME}")
    os.makedirs(save_dir, exist_ok=True)

    lora_configs: Dict[str, Any] = {
        "r": 4,
        "lora_alpha": 16,
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        "bias": "none", # none | all | lora_only
        "lora_dropout": 5e-2,
    }

    training_configs: Dict[str, Any] = {
        "output_dir": os.path.join(save_dir, f"lora-trace"),
        # Gradients related
        "gradient_accumulation_steps": 32,
        "max_grad_norm": 1.0,
        # Training related
        "fp16": True,
        "learning_rate": 2e-4,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "dataloader_pin_memory": True,
        # Saving & Evaluation related
        "logging_steps": 100,
        "eval_strategy": 'steps',
        "eval_steps": 100,
        "save_strategy": 'steps',
        "save_steps": 100,
        "save_total_limit": 5,
        "metric_for_best_model": "eval_loss",
        "load_best_model_at_end": True,
    }

    # For Llama-3.2-3B-Instruct
    # configs: GeneralConfigs = LlamaConfigs(
    #     MODEL_ID="meta-llama/Llama-3.2-3B-Instruct",
    #     DEVICE_MAP="auto",
    # )

    # save_dir: str = os.path.join(".", f"LoRA-{configs.MODEL_NAME}-{configs.DATASET_DICT_NAME}")
    # os.makedirs(save_dir, exist_ok=True)


    # configs: GeneralConfigs = LlamaConfigs(
    #     # Llama-3.2-1B-Instruct
    #     MODEL_ID="meta-llama/Llama-3.2-1B-Instruct",
    #     LORA_TARGET_MODULES=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    #     DEVICE_MAP="auto",

    #     # Llama-3.2-3B-Instruct
    #     # MODEL_ID="meta-llama/Llama-3.2-3B-Instruct",
    #     # LORA_TARGET_MODULES=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    #     # DEVICE_MAP="auto",

    #     # Llama-3.1-8B-Instruct
    #     # MODEL_ID="meta-llama/Llama-3.1-8B-Instruct",
    #     # LORA_TARGET_MODULES=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    #     # DEVICE_MAP="auto",
    # )

    fine_tune_model(configs=configs, save_dir=save_dir,
                    lora_configs=lora_configs, training_configs=training_configs)

