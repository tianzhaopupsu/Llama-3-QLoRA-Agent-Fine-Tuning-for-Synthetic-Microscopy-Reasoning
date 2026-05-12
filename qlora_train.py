##### QLoRA fine-tuning

!pip install -q transformers datasets peft bitsandbytes accelerate trl
from huggingface_hub import login

login()  # paste your HF token
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files="/content/drive/MyDrive/LLM_fine_tuning/dataset.jsonl",
    split="train"
)

print(dataset[0])
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
def format_example(example):
    messages = example["messages"]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}

dataset = dataset.map(format_example)
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)
from trl import SFTTrainer
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="/content/llama3-qlora-react",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,

    logging_steps=10,

    evaluation_strategy="steps",   # IMPORTANT
    eval_steps=10,                 # how often to evaluate

    save_steps=50,
    save_total_limit=2,

    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=lora_config,
    args=training_args,
    max_seq_length=1024
)
trainer.train()
logs = trainer.state.log_history
train_loss = []
eval_loss = []
train_steps = []
eval_steps = []

for log in logs:
    if "loss" in log and "eval_loss" not in log:
        train_loss.append(log["loss"])
        train_steps.append(log["step"])

    if "eval_loss" in log:
        eval_loss.append(log["eval_loss"])
        eval_steps.append(log["step"])


plt.figure(figsize=(8,5))

plt.plot(train_steps, train_loss, label="Train Loss")
plt.plot(eval_steps, eval_loss, label="Validation Loss")

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss (QLoRA - Llama 3 8B)")
plt.legend()

plt.show()
trainer.model.save_pretrained("/content/llama3_qlora_adapter")
tokenizer.save_pretrained("/content/llama3_qlora_adapter")
