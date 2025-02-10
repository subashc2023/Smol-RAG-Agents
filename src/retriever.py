from smolagents import Tool
from sentence_transformers import SentenceTransformer
import chromadb

class ChromaRetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant documentation based on your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, model: SentenceTransformer, collection, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.collection = collection

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        
        # Encode the query using our model
        query_embedding = self.model.encode([query])
        
        # Search using ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=5
        )
        
        # Format results
        documents = []
        for i, (doc_id, content) in enumerate(zip(results['ids'][0], results['documents'][0])):
            documents.append(f"\n===== Document {i+1} ({doc_id}) =====\n{content}")
            
        return "\nRetrieved documents:" + "".join(documents)
