from smolagents import CodeAgent, LiteLLMModel, DuckDuckGoSearchTool
from dotenv import load_dotenv
from embed import initialize_model, setup_chromadb, load_documents
from retriever import ChromaRetrieverTool
import os
import sys

def parse_args():
    args = sys.argv[1:]
    use_web = False
    query = "Who invented Hangul?"
    
    if args:
        if "-web" in args:
            use_web = True
            args.remove("-web")
        if args:
            query = " ".join(args)
            
    return use_web, query

def main():
    load_dotenv()
    model = LiteLLMModel(model_id="gemini/gemini-2.0-flash", api_key=os.getenv("GEMINI_API_KEY"))
    embedding_model = initialize_model()
    passages = load_documents()
    collection = setup_chromadb(embedding_model, passages)

    use_web, query = parse_args()
    tools = [ChromaRetrieverTool(model=embedding_model, collection=collection)]
    
    if use_web:
        tools.append(DuckDuckGoSearchTool())

    agent = CodeAgent(
        tools=tools,
        model=model,
        add_base_tools=True
    )

    agent.run(query)

if __name__ == "__main__":
    main()