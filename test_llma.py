import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Path to fine-tuned model
model_path = "resulter/final_model"

# Enable 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,  # More stable than float16
    bnb_4bit_use_double_quant=True
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding

# Test prompt (Concise input format)
test_prompt = "What is the capital of France?\nA:"

# Tokenize input
device = "cuda" if torch.cuda.is_available() else "cpu"
inputs = tokenizer(test_prompt, return_tensors="pt", add_special_tokens=True).to(device)

# Generate response (limit to single output)
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=50,  # Short answer
        temperature=0.3,  # Low randomness
        top_k=10,  # More deterministic
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id  # Stop generation at EOS token
    )

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Extract only the response (remove prompt)
response = generated_text.replace(test_prompt, "").strip()

print("Model Output:", response)
