from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

app = FastAPI()

# Load API keys (consider using environment variables for security)
openai.api_key = "your-openai-api-key"
anthropic = Anthropic(api_key="your-anthropic-api-key")

class AIRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"  # Default to GPT-3.5

# Read META_PROMPT from file
with open("meta_prompt.txt", "r") as f:
    META_PROMPT = f.read().strip()

@app.post("/process_request")
async def process_request(request: AIRequest):
    full_prompt = f"{META_PROMPT}\n\nCommunity Request: {request.prompt}\n\nYour response:"
    
    try:
        if request.model.startswith("gpt"):
            response = openai.ChatCompletion.create(
                model=request.model,
                messages=[{"role": "system", "content": META_PROMPT},
                          {"role": "user", "content": request.prompt}]
            )
            result = response.choices[0].message.content
        elif request.model == "claude":
            response = anthropic.completions.create(
                model="claude-2",
                prompt=f"{HUMAN_PROMPT} {full_prompt}{AI_PROMPT}",
                max_tokens_to_sample=300
            )
            result = response.completion
        else:
            raise HTTPException(status_code=400, detail="Unsupported model")
        
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "GOV AI microservice is running"}
