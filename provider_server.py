from fastapi import FastAPI, Request, HTTPException
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from redisvl.schema import IndexSchema
from llama_index.readers.google import GoogleDriveReader
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Global variables
vector_store = None
index = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    print(f"Headers: {request.headers}")
    body = await request.body()
    print(f"Body: {body}")
    response = await call_next(request)
    return response

def configure_settings():
    # Configure embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Configure Redis vector store
    custom_schema = IndexSchema.from_dict(
        {
            "index": {"name": "gdrive", "prefix": "doc"},
            "fields": [
                {"type": "tag", "name": "id"},
                {"type": "tag", "name": "doc_id"},
                {"type": "text", "name": "text"},
                {
                    "type": "vector",
                    "name": "vector",
                    "attrs": {
                        "dims": 384,
                        "algorithm": "hnsw",
                        "distance_metric": "cosine",
                    },
                },
            ],
        }
    )
    global vector_store
    vector_store = RedisVectorStore(
        schema=custom_schema, redis_url="redis://172.17.0.4:6379"
    )

def load_data_from_gdrive(folder_id: str):
    loader = GoogleDriveReader(redirect_uri=os.getenv("GDRIVE_REDIRECT_URI"))
    docs = loader.load_data(folder_id=folder_id)
    for doc in docs:
        print(doc.metadata)
        doc.id_ = doc.metadata["file name"]
    return docs

def create_index(docs):
    global index
    index = VectorStoreIndex.from_documents(
        docs,
        vector_store=vector_store
    )

@app.on_event("startup")
async def startup_event():
    try:
        configure_settings()
        docs = load_data_from_gdrive(folder_id=os.getenv("GDRIVE_FOLDER_ID"))
        create_index(docs)
    except Exception as e:
        print(f"Error during startup: {e}")

@app.post("/context-provider")
async def get_context(request: Request):
    try:
        data = await request.json()
        fullInput = data.get("fullInput")
        if not fullInput:
            raise HTTPException(status_code=400, detail="fullInput field is required")

        print(f"Received query: {fullInput}")

        retriever = index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(fullInput)

        # Create individual context items from retrieved nodes
        context_items = []
        for i, node in enumerate(nodes):
            context_items.append({
                "name": f"Result {i+1}",
                "description": f"Result {i+1} from Vector DB",
                "content": node.text
            })

        print(f"Query response: {context_items}")

        return context_items

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
