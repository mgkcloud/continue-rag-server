Create a RAG server which can take files from Gdrive and search them as RAG context for continue.dev
(Based on https://docs.llamaindex.ai/en/stable/examples/ingestion/ingestion_gdrive/)



```
pip3 install -r requirements.txt
```

ensure redis is up

```

# if creating a new container
!docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
# # if starting an existing container
# !docker start -a redis-stack

```


run server

```
uvicorn provider_server:app --host 0.0.0.0 --port 8027
```

In continue config 

``` 
{
      "name": "http",
      "params": {
        "url": "http://localhost:8027/context-provider",
        "title": "vector-db",
        "description": "Custom HTTP Context Provider",
        "displayTitle": "RAG"
      }
}
```