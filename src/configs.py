import torch

from src.utils import system_prompt


DATASET_DICT_ID: str = "openai/gsm8k"
DATASET_NAME: str = "main"
NUM_WORKERS: int = 4

MODEL_ID: str = "meta-llama/Llama-3.2-1B-Instruct"
TORCH_DTYPE: torch.dtype = torch.bfloat16
DEVICE_MAP: str = 'cuda:0'

DATASET_DICT_NAME: str = DATASET_DICT_ID.split("/")[-1]
MODEL_NAME: str = MODEL_ID.split("/")[-1]

OUTPUT_FORMAT: str = "The answer is: "
SYSTEM_PROMPT: str = system_prompt(
    f"You are a helpful assistant. Please solve the math problem step by step."
    f"Show your work clearly and state your final answer at the end like '{OUTPUT_FORMAT}XXX'."
)

BATCH_SIZE: int = 64
INPUT_MAX_LENGTH: int = 1024
MAX_NEW_TOKENS: int = 1024


