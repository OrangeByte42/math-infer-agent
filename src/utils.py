from typing import Tuple

import torch


def show_devices_info() -> None:
    print(f"--- GPU Availability ---")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPUs available.")
        return None
    print("")

    print(f"--- GPU Numbers ---")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("")

    print(f"--- VRAM ---")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i} VRAM: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
    print("")

    print(f"--- Supported Computing Precision ---")
    if torch.cuda.is_bf16_supported(): print("BF16 is supported.")

def calculate_duration_time(start_ts: float, end_ts: float) -> Tuple[int, int]:
    duration: int = int(end_ts - start_ts)
    minutes: int = duration // 60
    seconds: int = duration % 60
    return minutes, seconds

