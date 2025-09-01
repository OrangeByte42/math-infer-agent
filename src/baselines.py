import re
import json
import time
from typing import Any, List, Dict

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from src.configs.configs import GeneralConfigs, LlamaConfigs
from src.utils import show_devices_info, calculate_duration_time


def evaluate_in_math_infer_tasks(configs: GeneralConfigs) -> None:
    """ A baseline for math reasoning tasks using a pre-trained / fine-tuned language model."""

    def extract_pred_answer(generated_text: str) -> str:
        candidate_fragment: str = generated_text.split(configs.OUTPUT_FORMAT)[-1].strip()
        candidate_numbers: List[str] = re.findall(r"[+-]?(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d*|0)(?:\.\d+)?", candidate_fragment)
        pred_answers: str = candidate_numbers[-1].replace(',', '') if candidate_numbers else ""
        return pred_answers

    def collate_fn(batch: Any) -> Dict[str, Any]:
        questions = [item['question'] for item in batch]
        labels = [item['answer'].split('####')[-1].replace(',', '').strip() for item in batch]
        prompts = [configs.SYSTEM_PROMPT + configs.user_prompt(q) + configs.assistant_todo_prompt() for q in questions]
        return {'prompts': prompts, 'labels': labels, 'questions': questions}

    start_ts: float = time.time()

    # Show devices info
    print(f"=== Show Devices Info ===")
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
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    # Load Dataset & DataLoader
    print(f"=== Load Dataset & DataLoader ===")
    dataset_dict: DatasetDict = load_dataset(configs.DATASET_DICT_ID, name=configs.DATASET_NAME)
    test_dataset: Dataset = dataset_dict['test']
    data_loader: DataLoader = DataLoader(test_dataset, batch_size=configs.BATCH_SIZE,
                                            collate_fn=collate_fn, num_workers=configs.NUM_WORKERS)

    # Inference in dataset
    print(f"=== Inference in Dataset ===")
    right_num, total_num = 0, 0
    trace_records: List[Any] = []
    unusual_records: List[Any] = []

    model.eval()
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc='Evaluating'):
            # Prepare Batch
            prompts: List[str] = batch['prompts']
            label_answers: List[str] = batch['labels']
            questions: List[str] = batch['questions']

            # Prepare Inputs
            inputs: Any = tokenizer(prompts, return_tensors='pt', truncation=True,
                                    padding=True, max_length=configs.INPUT_MAX_LENGTH)
            inputs: Dict[str, Any] = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate Outputs
            outputs: Any = model.generate(
                **inputs,
                max_new_tokens=configs.MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            # Decode Outputs
            input_lengths: Any = inputs['input_ids'].shape[1]
            new_tokens: Any = outputs[:, input_lengths:]
            decoded_outputs: Any = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            pred_answers: List[str] = [extract_pred_answer(out) for out in decoded_outputs]

            # Statistics
            right_num += sum(1 for pred, label in zip(pred_answers, label_answers) if pred == label)
            total_num += len(label_answers)

            # Save trace
            for question, decoded_output, label_answer, pred_answer in zip(questions, decoded_outputs, label_answers, pred_answers):
                try:
                    trace_records.append({
                        'question': question,
                        'generated_answer': decoded_output,
                        'label_answer': label_answer,
                        'pred_answer': pred_answer,
                        'is_correct': float(pred_answer) == float(label_answer),
                    })
                except Exception:
                    record = {
                        'question': question,
                        'generated_answer': decoded_output,
                        'label_answer': label_answer,
                        'pred_answer': pred_answer,
                        'is_correct': False,
                    }
                    trace_records.append(record)
                    unusual_records.append(record)

        # Final Accuracy
        print(f"=== Final Accuracy ===")
        accuracy: float = right_num / total_num
        print(f"Final Accuracy: {accuracy*100:.4f}% ({right_num}/{total_num})")

        end_ts: float = time.time()
        minutes, seconds = calculate_duration_time(start_ts, end_ts)

        # Save Trace Records
        print(f"=== Save Inference Trace ===")
        runtime_trace = {
            "Running Configs": {
                "MODEL_ID": configs.MODEL_ID,
                "DATASET_ID": configs.DATASET_DICT_ID,
                "DATASET_NAME": configs.DATASET_NAME,
                "MODEL_DTYPE": str(configs.TORCH_DTYPE),
                "INPUT_MAX_LENGTH": configs.INPUT_MAX_LENGTH,
                "MAX_NEW_TOKENS": configs.MAX_NEW_TOKENS,
                "BATCH_SIZE": configs.BATCH_SIZE,
                "SYSTEM_PROMPT": configs.SYSTEM_PROMPT,
                "OUTPUT_FORMAT": configs.OUTPUT_FORMAT,
            },
            "Running Results": {
                "Total Samples Number": total_num,
                "Correct Samples Number": right_num,
                "Wrong Samples Number": total_num - right_num,
                "Final Accuracy": f"{accuracy*100:.4f}% ({right_num:,}/{total_num:,})",
            },
            "Running Time": {
                "Start Timestamp (s)": start_ts,
                "End Timestamp (s)": end_ts,
                "Duration (mm:ss)": f"{minutes}m:{seconds}s ({int(end_ts - start_ts):,} s)",
            },
            "Total Trace Records": trace_records,
            "Classification Trace Records": {
                "Correct": [record for record in trace_records if record['is_correct']],
                "Wrong": [record for record in trace_records if not record['is_correct']],
                "Unusual": unusual_records,
            }
        }

        with open(f"{configs.MODEL_NAME}-{configs.DATASET_DICT_NAME}-Runtime_Trace.jsonl", "w", encoding='utf-8') as f:
            json.dump(runtime_trace, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    configs: GeneralConfigs = LlamaConfigs(
        # Llama-3.2-1B-Instruct
        MODEL_ID="meta-llama/Llama-3.2-1B-Instruct",
        BATCH_SIZE=64,
        DEVICE_MAP="cuda:0",

        # Llama-3.2-3B-Instruct
        # MODEL_ID="meta-llama/Llama-3.2-3B-Instruct",
        # BATCH_SIZE=48,
        # DEVICE_MAP="auto",

        # Llama-3.1-8B-Instruct
        # MODEL_ID="meta-llama/Llama-3.1-8B-Instruct",
        # BATCH_SIZE=24,
        # DEVICE_MAP="auto",
    )
    configs.OUTPUT_FORMAT = "The final answer is: "
    configs.SYSTEM_PROMPT = configs.system_prompt(
        f"You are a helpful assistant. Please solve the math problem step by step."
        f"Show your work clearly and state your final answer at the end like '{configs.OUTPUT_FORMAT}XXX'."
    )

    evaluate_in_math_infer_tasks(configs=configs)

