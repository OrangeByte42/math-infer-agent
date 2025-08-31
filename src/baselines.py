import re
import json
from typing import List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm

from src.configs import *
from src.utils import show_devices_info
from src.utils import user_prompt, assistant_todo_prompt


def extract_pred_answer(generated_text: str) -> str:
    candidate_fragment: str = generated_text.split(OUTPUT_FORMAT)[-1].strip()
    candidate_numbers: List[str] = re.findall(r"[+-]?(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d*|0)(?:\.\d+)?", candidate_fragment)
    pred_answers: str = candidate_numbers[-1].replace(',', '') if candidate_numbers else ""
    return pred_answers

def collate_fn(batch):
    questions = [item['question'] for item in batch]
    labels = [item['answer'].split('####')[-1].strip() for item in batch]
    prompts = [SYSTEM_PROMPT + user_prompt(q) + assistant_todo_prompt() for q in questions]
    return {'prompts': prompts, 'labels': labels, 'questions': questions}


if __name__ == '__main__':
    print(f"=== Show Devices Info ===")
    show_devices_info()

    print(f"=== Set Seed & CUDNN Benchmark ===")
    torch.backends.cudnn.benchmark = True

    print(f"=== Load Model ===")
    model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        device_map=DEVICE_MAP,
    )

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    print(f"=== Load Dataset ===")
    ds: DatasetDict = load_dataset(DATASET_DICT_ID, name=DATASET_NAME)
    test_ds: Dataset = ds['test']
    data_loader: DataLoader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    print(f"=== Inference ===")
    right_num, total_num = 0, 0
    trace_records = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Prepare Batch
            prompts = batch['prompts']
            label_answers = batch['labels']
            questions = batch['questions']

            # Prepare Inputs
            inputs = tokenizer(prompts, return_tensors='pt', truncation=True,
                                padding=True, max_length=INPUT_MAX_LENGTH)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate Outputs
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
            )

            # Decode Outputs
            input_lengths = inputs['input_ids'].shape[1]
            new_tokens = outputs[:, input_lengths:]
            decoded_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            pred_answers = [extract_pred_answer(out) for out in decoded_outputs]

            # Statistics
            right_num += sum(1 for pred, label in zip(pred_answers, label_answers) if pred == label)
            total_num += len(label_answers)

            # Save trace
            for question, decoded_output, label_answer, pred_answer in zip(questions, decoded_outputs, label_answers, pred_answers):
                trace_records.append({
                    'question': question,
                    'generated_answer': decoded_output,
                    'label_answer': label_answer,
                    'pred_answer': pred_answer,
                    'is_correct': pred_answer == label_answer,
                })
    # Final Accuracy
    print(f"=== Final Accuracy ===")
    accuracy = right_num / total_num

    print(f"Final Accuracy: {accuracy*100:.4f}% ({right_num}/{total_num})")

    # Save Trace Records
    print(f"=== Save Inference Trace ===")
    runtime_trace = {
        "Running Configs": {
            "MODEL_ID": MODEL_ID,
            "DATASET_ID": DATASET_DICT_ID,
            "DATASET_NAME": DATASET_NAME,
            "MODEL_DTYPE": str(TORCH_DTYPE),
            "INPUT_MAX_LENGTH": INPUT_MAX_LENGTH,
            "MAX_NEW_TOKENS": MAX_NEW_TOKENS,
            "BATCH_SIZE": BATCH_SIZE,
            "SYSTEM_PROMPT": SYSTEM_PROMPT,
            "OUTPUT_FORMAT": OUTPUT_FORMAT,
        },
        "Running Results": {
            "Total Samples Number": total_num,
            "Correct Samples Number": right_num,
            "Wrong Samples Number": total_num - right_num,
            "Final Accuracy": f"{accuracy*100:.4f}% ({right_num}/{total_num})",
        },
        "Total Trace Records": trace_records,
        "Classification Trace Records": {
            "Correct": [record for record in trace_records if record['is_correct']],
            "Wrong": [record for record in trace_records if not record['is_correct']],
        }
    }

    with open(f"{MODEL_NAME}-{DATASET_DICT_NAME}-Runtime_Trace.jsonl", "w", encoding='utf-8') as f:
        json.dump(runtime_trace, f, ensure_ascii=False, indent=4)

