from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI()

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define request model
class ChatRequest(BaseModel):
    prompt: str

# Load fine-tuned model
model_path = "resulter/final_model"

# Enable 4-bit quantization for efficient inference
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load model
print("Loading model...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise RuntimeError("Failed to load model")

# Serve the frontend index.html
@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")

# Chatbot API endpoint
@app.post("/generate")
async def generate_response(request: ChatRequest):
    try:
        # Format the prompt
        test_prompt = request.prompt + "\nA:"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Tokenize input and move to device
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_length=30,
                temperature=0.3,
                top_k=10,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the response (remove prompt)
        response = generated_text.replace(test_prompt, "").strip()

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

# Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from fastapi import FastAPI, Request
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from fastapi.middleware.cors import CORSMiddleware

# # Define FastAPI app
# app = FastAPI()

# # Enable CORS for frontend requests
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins for testing; restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Path to fine-tuned model
# model_path = "result/final_model"

# # Enable 4-bit quantization
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )

# # Load model with quantization
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     quantization_config=bnb_config
# )

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# tokenizer.pad_token = tokenizer.eos_token  # Ensure correct padding

# # Define request model
# class PromptRequest(BaseModel):
#     prompt: str

# @app.post("/generate")
# async def generate_text(request: PromptRequest):
#     prompt_text = request.prompt.strip()
#     if not prompt_text:
#         return JSONResponse(content={"error": "Prompt cannot be empty"}, status_code=400)

#     # Tokenize input
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True).to(device)

#     # Generate response
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             max_length=50,
#             temperature=0.3,
#             top_k=10,
#             top_p=0.9,
#             do_sample=True,
#             eos_token_id=tokenizer.eos_token_id
#         )

#     # Decode output
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Extract only the response (remove prompt)
#     response = generated_text.replace(prompt_text, "").strip()

#     return {"response": response}
