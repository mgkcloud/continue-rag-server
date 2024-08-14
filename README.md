pip3 install -r requirements.txt


uvicorn provider_server:app --host 0.0.0.0 --port 8027


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