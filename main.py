from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import ast
from typing import Optional, List, Dict
import json
import importlib.util
import re

from En import (initialize_bot as initialize_en_bot, enhanced_chat_response as en_chat_response)
from Jp import (initialize_bot as initialize_jp_bot, enhanced_chat_response as jp_chat_response)

try:
    initialize_en_bot()
    initialize_jp_bot()
except Exception as e:
    print(f"Error initializing bots: {str(e)}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = 'en' 

class ChatResponse(BaseModel):
    response: str
    image_base64: Optional[str]
    confidence: str
    related_topics: list[str]

class DatasetEntry(BaseModel):
    question: str
    answer: str
    category: str
    image_path: Optional[str]
    related_topics: list[str]

class DatasetUpdateRequest(BaseModel):
    data: DatasetEntry
    language: str

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        if request.language == 'jp':
            print(f"Processing Japanese query: {request.message}")
            result = jp_chat_response(request.message)
            print(f"Japanese response: {result}")
        else:
            result = en_chat_response(request.message)
            
        return ChatResponse(
            response=result['response'],
            image_base64=result['image_base64'],
            confidence=result['confidence'],
            related_topics=result['related_topics']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_data")
async def get_data(language: str = "en"):
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'Jp.py' if language == 'jp' else 'En.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        match = re.search(r'business_data\s*=\s*(\[[\s\S]*?\n\])', content)
        if not match:
            raise HTTPException(status_code=400, detail="Could not find business_data in the file")
        
        data_str = match.group(1)
        business_data = ast.literal_eval(data_str)
        return business_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update_data")
async def update_dataset(request: DatasetUpdateRequest):
    try:
        file_path = os.path.join(os.path.dirname(__file__), 
                               'Jp.py' if request.language == 'jp' else 'En.py')
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        data_start = content.find('business_data = [')
        if data_start == -1:
            raise HTTPException(status_code=400, detail="Could not find business_data in the file")
        
        data_end = content.find(']', data_start)
        if data_end == -1:
            raise HTTPException(status_code=400, detail="Could not parse business_data")
        
        new_entry = f"""    {{
        "question": {json.dumps(request.data.question, ensure_ascii=False)},
        "answer": {json.dumps(request.data.answer, ensure_ascii=False)},
        "category": {json.dumps(request.data.category, ensure_ascii=False)},
        "image_path": {json.dumps(request.data.image_path, ensure_ascii=False)},
        "related_topics": {json.dumps(request.data.related_topics, ensure_ascii=False)}
    }},\n"""
        
        updated_content = (
            content[:data_start + len('business_data = [')] +
            '\n' + new_entry +
            content[data_start + len('business_data = ['):]
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        if request.language == 'jp':
            importlib.reload(sys.modules['Jp'])
            initialize_jp_bot()
        else:
            importlib.reload(sys.modules['En'])
            initialize_en_bot()
        
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}