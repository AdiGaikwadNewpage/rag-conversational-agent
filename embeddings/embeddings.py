from embeddings.openai_embeddings import OpenAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings

class Embeddings:
    def __init__(self, model_name: str, api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    def get_embedding_function(self):
        if self.model_name == "ollama":
            return OllamaEmbeddings(model="mxbai-embed-large")
        elif self.model_name == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key must be provided for OpenAI embeddings")
            return OpenAIEmbeddings(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported embedding model: {self.model_name}")