import os
import json
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training
)
from trl import SFTTrainer, SFTConfig 

# Environment settings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ==========================================
# 1. LOAD JSONL
# ==========================================
def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))
    return data

raw_data = load_jsonl("LLM_fine_tuning/train_dataset.jsonl")
dataset = Dataset.from_list(raw_data)
print("Sample raw data:")
print(dataset[0])

# ==========================================
# 2. MODEL CONFIG
# ==========================================
model_name = "meta-llama/Llama-3.2-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Kept float16 for RTX 4050 compatibility
    bnb_4bit_use_double_quant=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Crucial for causal language modeling packing

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model.config.use_cache = False

# ==========================================
# 3. PREPARE MODEL FOR QLORA
# ==========================================
model = prepare_model_for_kbit_training(model)

# ==========================================
# 4. FORMAT CHAT DATA (Only create the raw text string)
# ==========================================
def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_example)

# ==========================================
# 5. LORA CONFIG
# ==========================================
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

# ==========================================
# 6. TRAINING ARGS
# ==========================================
output_dir = "./llama3_qlora_output"

# SFTConfig replaces TrainingArguments and accepts dataset parameters directly
training_args = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=1,
    save_steps=100,
    save_total_limit=2,
    fp16=False,               
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit", 
    report_to="none",
    
    # --- Move these parameters here ---
    dataset_text_field="text", # Points to your mapped 'text' column
    max_length=512,        # Handles tokenization constraints internally
)

# ==========================================
# 7. TRAINER
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,        # Pass the unified SFTConfig here
)
# ==========================================
# 8. TRAIN
# ==========================================
print("\nStarting training...\n")
trainer.train()

# ==========================================
# 9. LOSS CURVE
# ==========================================
logs = trainer.state.log_history
train_loss = []
train_steps = []
for log in logs:
    if "loss" in log:
        train_loss.append(log["loss"])
        train_steps.append(log["step"])

if train_loss:
    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, train_loss, label="Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("QLoRA Training Curve")
    plt.legend()
    plt.show()
else:
    print("No loss history found to plot. Ensure logging_steps is less than total steps.")

# ==========================================
# 10. SAVE ADAPTER
# ==========================================
save_path = "./llama3_qlora_adapter"
trainer.model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("\nSaved adapter successfully to:")
print(save_path)
