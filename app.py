from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

# Load the AI model and tokenizer at startup
model_name = "facebook/bart-large-cnn"  # Example AI model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

class AIRequest(BaseModel):
    prompt: str

@app.post("/process_request")
async def process_request(request: AIRequest):
    inputs = tokenizer(request.prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": result}

@app.get("/")
async def read_root():
    return {"message": "GOV AI microservice is running"}
