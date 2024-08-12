import os
import logging
from fastapi import FastAPI, Request, HTTPException
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.readers.google import GoogleDriveReader
from lancedb.rerankers import ColbertReranker
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global variables
vector_store: LanceDBVectorStore = None
index: VectorStoreIndex = None

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.info(f"Headers: {request.headers}")
    body = await request.body()
    logger.info(f"Body: {body}")
    response = await call_next(request)
    return response

def configure_settings() -> None:
    try:
        # Configure embedding model to use a local model
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Configure LanceDB vector store with reranker
        global vector_store
        reranker = ColbertReranker()
        vector_store = LanceDBVectorStore(
            uri="./lancedb", 
            embed_model=embed_model,
            reranker=reranker, 
            mode="overwrite"
        )
        logger.info("Settings configured successfully.")
    except Exception as e:
        logger.error(f"Error configuring settings: {e}")
        raise

def load_data_from_gdrive(folder_id: str):
    try:
        loader = GoogleDriveReader(redirect_uri=os.getenv("GDRIVE_REDIRECT_URI"))
        docs = loader.load_data(folder_id=folder_id)
        for doc in docs:
            logger.info(doc.metadata)
            doc.id_ = doc.metadata.get("file name", doc.id_)
        return docs
    except Exception as e:
        logger.error(f"Error loading data from Google Drive: {e}")
        return []

def create_index(docs) -> None:
    try:
        global index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            docs,
            storage_context=storage_context
        )
        logger.info("Index created successfully.")
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    try:
        configure_settings()
        docs = load_data_from_gdrive(folder_id=os.getenv("GDRIVE_FOLDER_ID"))
        if docs:
            create_index(docs)
        else:
            logger.warning("No documents loaded from Google Drive.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.post("/context-provider")
async def get_context(request: Request):
    try:
        data = await request.json()
        fullInput = data.get("fullInput")
        if not fullInput:
            raise HTTPException(status_code=400, detail="fullInput field is required")

        logger.info(f"Received query: {fullInput}")

        if index is None:
            raise HTTPException(status_code=500, detail="Index is not initialized")

        # Use LanceDB filter for now, can integrate with Query engine later
        retriever = index.as_retriever(similarity_top_k=2, vector_store_kwargs={"where": "metadata.file_name IS NOT NULL"})
        nodes = retriever.retrieve(fullInput)

        # Create individual context items from retrieved nodes
        context_items = [
            {
                "name": f"Result {i+1}",
                "description": f"Result {i+1} from Vector DB",
                "content": node.text
            }
            for i, node in enumerate(nodes)
        ]

        logger.info(f"Query response: {context_items}")

        return context_items

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
