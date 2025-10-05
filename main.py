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

# Initialize the chatbot and knowledge base
from En import (initialize_bot as initialize_en_bot, enhanced_chat_response as en_chat_response)
from Jp import (initialize_bot as initialize_jp_bot, enhanced_chat_response as jp_chat_response)

# Initialize both bots
try:
    initialize_en_bot()
    initialize_jp_bot()
except Exception as e:
    print(f"Error initializing bots: {str(e)}")
    # Continue anyway, bots will be initialized on first use

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    language: str = 'en'  # Default to English

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
        # Use explicitly specified language instead of detection
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

@app.post("/api/update_data")
async def update_dataset(request: DatasetUpdateRequest):
    try:
        # Determine which file to update based on language
        file_path = os.path.join(os.path.dirname(__file__), 
                               'Jp.py' if request.language == 'jp' else 'En.py')
        
        # Read the current file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find the business_data list
        data_start = content.find('business_data = [')
        if data_start == -1:
            raise HTTPException(status_code=400, detail="Could not find business_data in the file")
        
        # Parse the existing data
        data_end = content.find(']', data_start)
        if data_end == -1:
            raise HTTPException(status_code=400, detail="Could not parse business_data")
        
        # Convert the new entry to proper format
        new_entry = f"""    {{
        "question": "{request.data.question}",
        "answer": "{request.data.answer}",
        "category": "{request.data.category}",
        "image_path": "{request.data.image_path}",
        "related_topics": {json.dumps(request.data.related_topics, ensure_ascii=False)}
    }},\n"""
        
        # Insert the new entry at the beginning of the data list
        updated_content = (
            content[:data_start + len('business_data = [')] +
            '\n' + new_entry +
            content[data_start + len('business_data = ['):]
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        # Reinitialize the appropriate bot
        if request.language == 'jp':
            initialize_jp_bot()
        else:
            initialize_en_bot()
            
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get_data")
async def get_data(language: str = "en"):
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'Jp.py' if language == 'jp' else 'En.py')
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract business_data list using regex
        match = re.search(r'business_data\s*=\s*(\[[\s\S]*?\n\])', content)
        if not match:
            raise HTTPException(status_code=400, detail="Could not find business_data in the file")
        
        # Safely evaluate the Python list literal
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
        
        # Find the business_data list
        data_start = content.find('business_data = [')
        if data_start == -1:
            raise HTTPException(status_code=400, detail="Could not find business_data in the file")
        
        # Parse the existing data
        data_end = content.find(']', data_start)
        if data_end == -1:
            raise HTTPException(status_code=400, detail="Could not parse business_data")
        
        # Format new entry with proper indentation and escaping
        new_entry = f"""    {{
        "question": {json.dumps(request.data.question, ensure_ascii=False)},
        "answer": {json.dumps(request.data.answer, ensure_ascii=False)},
        "category": {json.dumps(request.data.category, ensure_ascii=False)},
        "image_path": {json.dumps(request.data.image_path, ensure_ascii=False)},
        "related_topics": {json.dumps(request.data.related_topics, ensure_ascii=False)}
    }},\n"""
        
        # Insert the new entry at the beginning of the data list
        updated_content = (
            content[:data_start + len('business_data = [')] +
            '\n' + new_entry +
            content[data_start + len('business_data = ['):]
        )
        
        # Write the updated content back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        # Reload the module to update the running instance
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