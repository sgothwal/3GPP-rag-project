from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from Backend.pipeline import pipeline


app = FastAPI(title="RAG_PROJECT")
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    filename:list[str]
   


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        answer,filename= pipeline(request.query)

        
    
        return QueryResponse(answer=answer,filename=filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))