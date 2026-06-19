from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from inference import generate_response
from config import PORT

app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 20   # 去掉 Optional，直接给默认值
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    prompt = request.messages[-1].content   # 取最后一条消息的内容
    response_text = generate_response(prompt, max_tokens=request.max_tokens)
    print(f"DEBUG: response_text = '{response_text}'")
    return {
        "choices": [{"message": {"role": "assistant", "content": response_text}}]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)