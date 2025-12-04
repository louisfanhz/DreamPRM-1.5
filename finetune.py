import json
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from data import MyDataset, read_json
from model import compute_supervised_loss
from utils import input_processing, generate_target, set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm


def finetune(
    train_json_file="./data/finetune.json",
    model_path="OpenGVLab/InternVL3-1B",
    output_path="./weights",
    lr=5e-5,
    epochs=3,
    batch_size=1,
    max_patch_num=6,
    seed=42,
):
    set_seed(seed)
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        use_flash_attn=False,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    # Build dataloader using existing MyDataset
    train_dataset = MyDataset(read_json(train_json_file), max_patch_num)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            prompt, pixel_values, _ = batch
            prompt, pixel_values = prompt[0], pixel_values[0]
            
            input_ids, attention_mask, image_flags, pixel_values = input_processing(model, tokenizer, prompt, pixel_values)
            
            output = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
            ).logits
            
            target = generate_target(input_ids, tokenizer, model.template)
            loss = compute_supervised_loss(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    model.save_pretrained(output_path)
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    finetune()
