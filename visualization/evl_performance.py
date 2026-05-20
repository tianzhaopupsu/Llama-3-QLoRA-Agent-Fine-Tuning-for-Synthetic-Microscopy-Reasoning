import torch
import gradio as gr
import time
import threading

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ===================================
# CONFIG
# ===================================
BASE_MODEL = "meta-llama/Llama-3.2-8B-Instruct"
LORA_PATH = "./llama3_qlora_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================
# LOAD TOKENIZER
# ===================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

# ===================================
# LOAD BASE MODEL (4-bit)
# ===================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()


# ===================================
# GENERATION (FIXED SLICING)
# ===================================
def generate_response(message, use_adapter=True):
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if use_adapter:
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        else:
            with model.disable_adapter():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

    # Slices tokens correctly so text does not disappear
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ===================================
# GRADIO FUNCTION (STREAMING WORDS)
# ===================================
def process_demo(user_input, base_history, ft_history):
    base_out = generate_response(user_input, use_adapter=False)
    ft_out = generate_response(user_input, use_adapter=True)

    base_words = base_out.split(" ")
    ft_words = ft_out.split(" ")

    base_history.append({"role": "user", "content": user_input})
    base_history.append({"role": "assistant", "content": ""})

    ft_history.append({"role": "user", "content": user_input})
    ft_history.append({"role": "assistant", "content": ""})

    yield "", base_history, ft_history

    max_words = max(len(base_words), len(ft_words))
    current_base = []
    current_ft = []

    for i in range(max_words):
        if i < len(base_words):
            current_base.append(base_words[i])
        if i < len(ft_words):
            current_ft.append(ft_words[i])

        base_history[-1]["content"] = " ".join(current_base)
        ft_history[-1]["content"] = " ".join(current_ft)

        time.sleep(0.05)
        yield "", base_history, ft_history


# ===================================
# JAVASCRIPT SCREEN RECORDER (SECURE)
# ===================================
RECORD_JS = """
<script>
let mediaRecorder;
let recordedChunks = [];
let localStream;

async function startTabRecording() {
    try {
        // Safe, native browser popup requesting access
        localStream = await navigator.mediaDevices.getDisplayMedia({
            video: { displaySurface: "browser" }, 
            audio: false
        });
        
        // Use a widely accepted video container codec format
        mediaRecorder = new MediaRecorder(localStream, { mimeType: 'video/webm;codecs=vp9' });
        recordedChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'gradio_interface_demo.webm';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            // Cleanly close out the video stream track lines
            localStream.getTracks().forEach(track => track.stop());
            alert("Video download successfully pushed to your system Downloads folder!");
        };

        mediaRecorder.start();
        alert("🔴 RECORDING STARTED SYSTEM-WIDE! Send your prompts now.");
    } catch (err) {
        console.error("Error starting capture: ", err);
        alert("Recording permission denied or unsupported browser environment.");
    }
}

function stopTabRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    } else {
        alert("❌ Error: No active recording session running. Click 'Start Interface Recording' first.");
    }
}
</script>
"""


# ===================================
# GRADIO UI
# ===================================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    gr.HTML(RECORD_JS)

    gr.Markdown("# 🚀 Base vs Fine-Tuned LLM Comparison")

    # Interactive Toolkit Bar
    with gr.Row():
        start_rec_btn = gr.Button("🔴 Start Interface Recording", variant="primary")
        stop_rec_btn = gr.Button("⏹️ Stop & Save Video File", variant="stop")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ❌ Base Model")
            base_chatbot = gr.Chatbot()

        with gr.Column():
            gr.Markdown("### 🎯 Fine-Tuned Model")
            ft_chatbot = gr.Chatbot()

    user_msg = gr.Textbox(
        label="Input Telemetry",
        placeholder="Region: cluster\nPopulation: 8.5%\nInteraction STD: 0.75"
    )

    # Event Mapping Layout
    start_rec_btn.click(fn=None, js="() => { startTabRecording(); }")
    stop_rec_btn.click(fn=None, js="() => { stopTabRecording(); }")

    user_msg.submit(
        process_demo,
        inputs=[user_msg, base_chatbot, ft_chatbot],
        outputs=[user_msg, base_chatbot, ft_chatbot]
    )


# ===================================
# RUN APP
# ===================================
if __name__ == "__main__":
    demo.launch(inbrowser=True)
