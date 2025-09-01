from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List

import torch
from peft import TaskType


@dataclass
class GeneralConfigs(ABC):
    """Abstract configuration class."""

    # Dataset parameters
    DATASET_DICT_ID: str
    DATASET_NAME: str
    NUM_WORKERS: int

    # Model parameters
    MODEL_ID: str
    TORCH_DTYPE: torch.dtype
    DEVICE_MAP: str

    # Training parameters
    BATCH_SIZE: int
    INPUT_MAX_LENGTH: int
    MAX_NEW_TOKENS: int

    # Evaluation parameters
    SYSTEM_PROMPT: str
    OUTPUT_FORMAT: str

    # Saving parameters
    MODEL_NAME: str
    DATASET_DICT_NAME: str

    @abstractmethod
    def system_prompt(self: Any, prompt: str) -> str:
        ...

    @abstractmethod
    def user_prompt(self: Any, prompt: str) -> str:
        ...

    @abstractmethod
    def assistant_todo_prompt(self: Any) -> str:
        ...


@dataclass
class LlamaConfigs(GeneralConfigs):
    """Configuration for LLaMA model."""

    # Dataset parameters
    DATASET_DICT_ID: str = "openai/gsm8k"
    DATASET_NAME: str = "main"
    NUM_WORKERS: int = 4

    # Model parameters
    MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"
    TORCH_DTYPE: torch.dtype = torch.bfloat16
    DEVICE_MAP: str = "cuda:0"

    # Training parameters
    BATCH_SIZE: int = 64
    INPUT_MAX_LENGTH: int = 1024
    MAX_NEW_TOKENS: int = 1024

    # Evaluation parameters
    SYSTEM_PROMPT: Optional[str] = None
    OUTPUT_FORMAT: Optional[str] = None

    # Saving parameters
    MODEL_NAME: Optional[str] = None
    DATASET_DICT_NAME: Optional[str] = None

    # LoRA parameters
    LORA_R: int = 4
    LORA_ALPHA: int = 16
    LORA_DROPOUT: float = 5e-2
    LORA_TARGET_MODULES: Optional[List[str]] = None # ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    LORA_BIAS: str = "none" # none | all | lora_only
    LORA_TASK_TYPE: TaskType = TaskType.CAUSAL_LM

    # Special tokens
    EOT_TOKEN: str = "<|eot_id|>"

    def __post_init__(self: Any) -> None:
        self.MODEL_NAME = self.MODEL_ID.split("/")[-1]
        self.DATASET_DICT_NAME = self.DATASET_DICT_ID.split("/")[-1]

    def system_prompt(self: Any, prompt: str) -> str:
        return f"<|start_header_id|>system<|end_header_id|>{prompt}{self.EOT_TOKEN}"

    def user_prompt(self: Any, prompt: str) -> str:
        return f"<|start_header_id|>user<|end_header_id|>{prompt}{self.EOT_TOKEN}"

    def assistant_todo_prompt(self: Any) -> str:
        return f"<|start_header_id|>assistant<|end_header_id|>"

