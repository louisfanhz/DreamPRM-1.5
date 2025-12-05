import json
from transformers import AutoModel, AutoTokenizer
from data import build_test_dataloader
import torch
from utils import select_best_answer
from tqdm import tqdm


def print_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))

def run_prm():
    test_dataloader = build_test_dataloader(test_json_file = "./data/test-gemini-3-pro-preview-CharXiv-reasoning-results-dreamprm.json")

    MODEL_PATH = "OpenGVLab/InternVL3-1B"
    model = AutoModel.from_pretrained(
                MODEL_PATH,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                use_flash_attn=False,
            ).cuda()
    tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL3-1B", trust_remote_code=True, use_fast=False)

    total = 0
    for inputs in tqdm(test_dataloader):
        # input_test_data_format:
        # {"question": question, "image_path": image_path, "candidate":[1, 2, 3, 4], "true_false":[True, False, True, False]}
        with torch.no_grad():
            select_best_answer(model, tokenizer, inputs, 'mean')
        total += 1

        if total >= 20:
            break

if __name__ == "__main__":
    run_prm()