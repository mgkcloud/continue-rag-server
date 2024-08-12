pip3 install -r requirements.txt


uvicorn provider_server:app --host 0.0.0.0 --port 8027

OR

uvicorn provider_lance:app --host 0.0.0.0 --port 8027



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

or in config.ts 

```
import { Config, ContextItem, ContextProviderExtras } from "continue";

const VectorDBContextProvider = {
  title: "vector-db",
  displayTitle: "Vector DB",
  description: "Retrieve snippets from our vector database",
  getContextItems: async (
    query: string,
    extras: ContextProviderExtras
  ): Promise<ContextItem[]> => {
    const response = await fetch(
      "https://localhost:8027/context-provider",
      {
        method: "POST",
        body: JSON.stringify({ query }),
      }
    );
    const results = await response.json();

    return results.map((result: any) => ({
      name: result.title,
      description: result.title,
      content: result.contents,
    }));
  },
};

export function modifyConfig(config: Config): Config {
  if (!config.contextProviders) {
    config.contextProviders = [];
  }
  config.contextProviders.push(VectorDBContextProvider);
  return config;
}

```