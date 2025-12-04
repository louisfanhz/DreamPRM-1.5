import json
from typing import Optional


def concatenate_json_files(
    file1: str,
    file2: str,
    output_file: str,
) -> None:
    # Load first JSON file
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    # Load second JSON file
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # Concatenate the two lists
    combined_data = data1 + data2
    
    # Write to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)
    

if __name__ == "__main__":
    # Concatenate train.json and cold_start.json
    concatenate_json_files(
        file1="data/train.json",
        file2="data/cold_start.json",
        output_file="data/cold_start_train.json"
    )

