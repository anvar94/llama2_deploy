from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import torch

# Load dataset
dataset = load_dataset("json", data_files="country_finetune.json")

# Enable 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load LLaMA-2 model
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# Load tokenizer and fix padding token
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Apply LoRA
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Freeze all parameters in the base model
for param in model.parameters():
    param.requires_grad = False

# Unfreeze LoRA parameters
for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True

# Disable gradient checkpointing for debugging
# model.gradient_checkpointing_enable()
model.config.use_cache = False  

# Tokenize dataset
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./resulter",  # Ensure this directory exists and is writable
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=1,  
    per_device_eval_batch_size=1,  
    gradient_accumulation_steps=8,  
    optim="adamw_bnb_8bit",  
    bf16=True,  # Use bfloat16 for stability
    num_train_epochs=3,  
    logging_steps=10,
    save_total_limit=2,  # Keep only the last 2 checkpoints
    weight_decay=0.01,
)

# Custom Trainer Class to Handle Loss
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# Use the custom trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["train"],
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Explicitly save the model
trainer.save_model("./resulter/final_model")