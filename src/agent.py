from smolagents import CodeAgent, LiteLLMModel
from dotenv import load_dotenv
from embed import initialize_model, setup_chromadb, load_documents
from retriever import ChromaRetrieverTool
import os

load_dotenv()

# Initialize components
model = LiteLLMModel(model_id="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = initialize_model()
passages = load_documents()
collection = setup_chromadb(embedding_model, passages)

# Initialize tools
#search_tool = DuckDuckGoSearchTool()
retriever_tool = ChromaRetrieverTool(model=embedding_model, collection=collection)

# Create agent with both tools
agent = CodeAgent(
    tools=[retriever_tool], 
    model=model, 
    #add_base_tools=True
)

# Example usage
if __name__ == "__main__":
    agent.run("Who invented Hangul?")