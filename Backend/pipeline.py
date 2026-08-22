from google import genai
from langchain_qdrant import QdrantVectorStore,FastEmbedSparse,RetrievalMode
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai import types
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel,Field
from typing import List,Optional
import cohere
import os


load_dotenv()
dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device':'mps'})
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/minicoil-v1",kwargs={'device': 'mps'})
co=cohere.ClientV2()
client = QdrantClient(url="http://localhost:6333")
collection_name="MAVENIR_PROJECT"

vector_store = QdrantVectorStore.from_existing_collection(
            embedding=dense_embeddings,
            sparse_embedding=sparse_embeddings,
            url='http://localhost:6333',
            collection_name=collection_name,
            retrieval_mode=RetrievalMode.HYBRID
            
        )



def retrieval(user_query)->list:
    results = vector_store.similarity_search_with_score(user_query,k=15)
    results=[result for result in results if result[1]>0.2]

    return(results)    



def rerank(chunks,user_query)->list:
   
    text_passage=[chunk[0].page_content for chunk in chunks]
    response = co.rerank(
    model="rerank-english-v3.0",
    query=user_query,
    documents=text_passage,
    top_n=8,
)
    top_text = [chunks[r.index] for r in response.results]
    return top_text 

def genereate(chunks,user_query)->str:
    
    class RAGResponse(BaseModel):
        answer: str=Field(description="answer to question")
        file_name:list[str]=Field(description="name of file(s) used for answering the question.Must be an empty list if no answer was found.")
    context = "\n\n\n".join([
    f"Page Content:{result[0].page_content}\n"
    + (f"Tables (HTML):{'\n'.join(result[0].metadata['table'])}\n" if result[0].metadata.get('table') else "")
    + f"File Location:{result[0].metadata['filename']}\n"
    for result in chunks
])
    SYSTEM_PROMPT = f"""
You are a helpful AI assistant who answers user query based on available context, retrieved from .docx file. You're provided with chunk text and table in html format if any
You should ONLY answer the user based on the following context and help user navigate to the right element, ALWAYS mention file name, IF you find no chunk matching the query ONLY send sorry but you didn't find anything, **and set file_name to an empty list in that case since no chunk was actually used**
User Query {user_query} \n
Context
{context}
"""
   
    llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")
    structred_llm=llm.with_structured_output(RAGResponse)
    response=structred_llm.invoke(SYSTEM_PROMPT)
    
    print(response)
    
    
    return(response.answer,response.file_name)

def pipeline(user_query):
    chunks=retrieval(user_query)
    rerank_chunks=rerank(chunks,user_query)
    ans=genereate(rerank_chunks,user_query)
    return(ans)
if __name__== "__main__":
    pipeline(" What are the specific technical requirements for implementing 5G New Radio unlicensed (NR-U) spectrum access in accordance with Release 16 specifications?")
