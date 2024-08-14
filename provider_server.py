import logging
import os
import asyncio
from typing import List
from contextlib import asynccontextmanager
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.readers.google import GoogleDriveReader
from llama_index.core.ingestion import IngestionPipeline, IngestionCache, DocstoreStrategy
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.core.node_parser import SentenceSplitter
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from redisvl.schema import IndexSchema

@lru_cache()
def get_env_variables():
    load_dotenv(override=True)
    return {
        "GDRIVE_REDIRECT_URI": os.getenv("GDRIVE_REDIRECT_URI"),
        "GDRIVE_FOLDER_ID": os.getenv("GDRIVE_FOLDER_ID"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "REDIS_URL": os.getenv("REDIS_URL"),
    }

get_env_variables.cache_clear()
env_vars = get_env_variables()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

vector_store = None
index = None
pipeline = None
flow = None

def configure_settings():
    global vector_store, pipeline
    
    embed_model = GeminiEmbedding(
        model_name="models/embedding-001",
        api_key=env_vars["GOOGLE_API_KEY"]
    )
    Settings.embed_model = embed_model
    logger.info("Gemini embedding model configured")

    redis_url = env_vars["REDIS_URL"]
    
    # Define the schema for the Redis vector store
    custom_schema = IndexSchema.from_dict({
        "index": {"name": "gdrive", "prefix": "doc"},
        "fields": [
            {"type": "tag", "name": "id"},
            {"type": "tag", "name": "doc_id"},
            {"type": "text", "name": "text"},
            {
                "type": "vector",
                "name": "vector",
                "attrs": {
                    "dims": 768,  # Gemini embedding dimension
                    "algorithm": "hnsw",
                    "distance_metric": "cosine",
                },
            },
        ],
    })

    # Create RedisVectorStore
    vector_store = RedisVectorStore(
        schema=custom_schema,
        redis_url=redis_url,
        overwrite=True
    )
    logger.info(f"Vector store configured with Redis URL: {redis_url}")

    cache = IngestionCache(
        cache=RedisCache.from_host_and_port("localhost", 6379),
        collection="redis_cache",
    )

    pipeline = IngestionPipeline(
        transformations=[SentenceSplitter(), embed_model],
        docstore=RedisDocumentStore.from_host_and_port("localhost", 6379, namespace="document_store"),
        vector_store=vector_store,
        cache=cache,
        docstore_strategy=DocstoreStrategy.UPSERTS,
    )
    logger.info("Ingestion pipeline configured")

def authenticate_google_drive():
    global flow
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json')
    if not creds or not creds.valid:
        flow = Flow.from_client_secrets_file(
            'credentials.json',
            scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        flow.redirect_uri = env_vars["GDRIVE_REDIRECT_URI"]
        authorization_url, _ = flow.authorization_url(prompt='consent')
        logger.info(f"Please go to this URL to authorize the application: {authorization_url}")
        return None
    return creds

async def load_data_from_gdrive() -> List[Document]:
    creds = authenticate_google_drive()
    if not creds:
        logger.error("Authentication required. Please run the server and complete the OAuth flow.")
        return []
    
    loader = GoogleDriveReader(credentials=creds)
    try:
        logger.info(f"Attempting to load data from Google Drive folder: {env_vars['GDRIVE_FOLDER_ID']}")
        all_files = await asyncio.to_thread(loader.load_data, folder_id=env_vars["GDRIVE_FOLDER_ID"])
        logger.info(f"Successfully loaded {len(all_files)} files from Google Drive")
    except Exception as e:
        logger.error(f"Error loading data from Google Drive: {str(e)}", exc_info=True)
        return []
    
    if not all_files:
        logger.warning("No files found in the specified Google Drive folder")
        return []
    
    for doc in all_files:
        doc.id_ = doc.metadata["file name"]
    
    return all_files

async def create_index(docs: List[Document]):
    global index
    try:
        logger.info("Creating new index...")
        nodes = pipeline.run(documents=docs)
        logger.info(f"Ingested {len(nodes)} Nodes")
        index = VectorStoreIndex.from_vector_store(pipeline.vector_store, embed_model=Settings.embed_model)
        logger.info(f"Index created successfully with {len(docs)} documents")
    except Exception as e:
        logger.error(f"Error creating index: {e}", exc_info=True)
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        configure_settings()
        docs = await load_data_from_gdrive()
        if docs:
            await create_index(docs)
        else:
            logger.warning("No documents loaded. Index creation skipped.")
    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
    
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/callback")
async def oauth2callback(request: Request):
    global flow
    code = request.query_params.get("code")
    flow.fetch_token(code=code)
    creds = flow.credentials
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    return "Authentication successful! You can close this window."

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

        # Aggregate content from retrieved nodes
        aggregated_content = "\n\n".join([node.text for node in nodes])

        # Return a single object with aggregated content
        response_data = {
            "name": "Context from VectorDB",
            "description": "Aggregated results from Vector DB",
            "content": aggregated_content
        }

        print(f"Query response: {response_data}")

        return response_data

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8027)
