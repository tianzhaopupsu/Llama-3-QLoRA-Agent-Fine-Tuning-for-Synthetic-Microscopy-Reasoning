####  UI to show
!pip install -q streamlit pyngrok transformers peft bitsandbytes accelerate
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import streamlit as st
import random
import time
from peft import PeftModel
from pyngrok import ngrok
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    device_map="auto"
)

# QLoRA model (your adapter)


qlora_model = PeftModel.from_pretrained(
    base_model,
    "/content/llama3_qlora_adapter"
)


class ToyEnv:
    def step(self):
        time.sleep(0.3)  # simulate microscope delay

        return {
            "region": random.choice(["single_cell", "cluster", "tissue_like"]),
            "population_pct": random.uniform(0, 100),
            "interaction_std": random.uniform(0, 1)
        }
def run_model(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.7
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
def compare_models(prompt):
    base_out = run_model(base_model, prompt)
    qlora_out = run_model(qlora_model, prompt)

    return base_out, qlora_out


st.title("Llama 3 QLoRA Agent Comparison")

env = ToyEnv()

prompt = st.text_area("Input prompt", "Analyze current microscopy region")

if st.button("Run Comparison"):

    base_out, qlora_out = compare_models(prompt)

    col1, col2 = st.columns(2)

    with col1:
        st.header("Base Llama 3")
        st.text(base_out)

    with col2:
        st.header("QLoRA Llama 3")
        st.text(qlora_out)


!streamlit run app.py &

public_url = ngrok.connect(8501)
print(public_url)
