from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from utils import get_output, chat_questions

app = FastAPI(
    title="News Summarization & Analysis",
    description="api.py file for routing requets between my functions",
    version="1.0.0"
)

DATA_DIR = "data"
OUTPUT_PATH = os.path.join(DATA_DIR, "output.json")
os.makedirs(DATA_DIR, exist_ok=True)

# the class Config is set with placeholder values in it's dict so that updating and assigning data from input forms and buttons would be easier

class CompanyRequest(BaseModel):
    company: str
    num_articles: int = 5
    
    class Config:
        schema_extra = {
            "example": {
                "company": "Microsoft",
                "num_articles": 5
            }
        }

class ChatRequest(BaseModel):
    question: str
    mode: str = "simple"
    corpus: str
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the main products of the company?",
                "mode": "simple",
                "corpus": "Sample text about the company..."
            }
        }

#below are routes to use the functions I defined in utils.py

# processes articles and gives json report about the company
@app.post("/process")
def process_company(request: CompanyRequest):
    try:
        output = get_output(request.company, request.num_articles)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=4)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# uses chat function for user q and a , either simple API call or RAG based method
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    try:
        answer = chat_questions(request.question, request.corpus, mode=request.mode)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running correctly"}

