import json
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore,FastEmbedSparse,RetrievalMode
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path

import os
load_dotenv()
dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device':'mps'})
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/minicoil-v1",kwargs={'device': 'mps'})
collection_name="MAVENIR_PROJECT"


def create_docu()-> list:
    try:
        base_path = Path(__file__).parent

        with open(base_path.parent / 'final_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: The file final_data.json does not exist.")  
    
    docu_list=[]
    for element in data:
        element['metadata']['tables']=element['table']
        doc=Document(page_content=element['text'],  ##langchain documents creation
                    
                metadata=element['metadata']
                )
        docu_list.append(doc)

  
    return(docu_list)
def create_embeddings(documents)->object:
    
    vector_store = QdrantVectorStore.from_documents(
    documents=documents,
    url='http://localhost:6333',
    embedding=dense_embeddings,
    sparse_embedding=sparse_embeddings,
    collection_name=collection_name,
    retrieval_mode=RetrievalMode.HYBRID
    
)
    print("!!!!Created vector store!!!!")
        
if __name__== "__main__":
    
     docs=create_docu()
     create_embeddings(docs)