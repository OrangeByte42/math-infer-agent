import torch


def show_devices_info():
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

def system_prompt(prompt: str) -> str:
    return f"<|start_header_id|>system<|end_header_id|>{prompt}<|eot_id|>"

def user_prompt(prompt: str) -> str:
    return f"<|start_header_id|>user<|end_header_id|>{prompt}<|eot_id|>"

def assistant_todo_prompt() -> str:
    return f"<|start_header_id|>assistant<|end_header_id|>"

