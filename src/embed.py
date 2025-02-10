from sentence_transformers import SentenceTransformer
import chromadb
import sys
import os
import glob
import hashlib
import json
from transformers import logging
from text_chunker import chunk_markdown

logging.set_verbosity_error()

def initialize_model():
    return SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

def load_documents():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docs_dir = os.path.join(project_root, "docs")
    passages = []
    chunk_info = []
    
    for md_file in glob.glob(os.path.join(docs_dir, "*.md")):
        basename = os.path.splitext(os.path.basename(md_file))[0]
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            chunks = chunk_markdown(content)
            for i, chunk in enumerate(chunks, start=1):
                passages.append(chunk)
                chunk_info.append((basename, i))
    
    return passages, chunk_info

def get_content_hash(passages):
    content = '\n'.join(sorted(passages))
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_hash_record():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hash_file = os.path.join(project_root, "chroma_db", "content_hash.json")
    if os.path.exists(hash_file):
        with open(hash_file, 'r') as f:
            return json.load(f)
    return {"content_hash": None}

def save_hash_record(content_hash):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hash_file = os.path.join(project_root, "chroma_db", "content_hash.json")
    os.makedirs(os.path.dirname(hash_file), exist_ok=True)
    with open(hash_file, 'w') as f:
        json.dump({"content_hash": content_hash}, f)

def setup_chromadb(model, passages):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_directory = os.path.join(project_root, "chroma_db")
    
    if passages:
        doc_texts, chunk_info = passages
    else:
        doc_texts, chunk_info = [], []
    
    current_hash = get_content_hash(doc_texts)
    stored_hash = load_hash_record()["content_hash"]
    
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="document_passages")
    
    if current_hash != stored_hash:
        existing_data = collection.get()
        existing_docs = existing_data.get("documents", [])
        existing_ids = existing_data.get("ids", [])
        
        existing_docs_set = set(existing_docs)
        new_passages = [(p, info) for p, info in zip(doc_texts, chunk_info) if p not in existing_docs_set]
        
        if new_passages:
            start_idx = len(existing_ids)
            new_texts = [p for p, _ in new_passages]
            new_info = [info for _, info in new_passages]
            
            passage_embeddings = model.encode(new_texts)
            collection.add(
                embeddings=passage_embeddings.tolist(),
                documents=new_texts,
                ids=[
                    f"passage_{global_index} - {info[0]}/chunk_{info[1]}"
                    for global_index, info in enumerate(new_info, start=start_idx + 1)
                ],
            )
            save_hash_record(current_hash)
            print(f"Added {len(new_passages)} new documents to ChromaDB.")
        else:
            print("No new documents to add.")
    else:
        print("No changes in documents detected, using existing vectors.")
    
    return collection

def perform_search(model, collection, query):
    task = 'Given a question, retrieve Wikipedia passages that answer the question'
    prompt = f"Instruct: {task}\nQuery: "
    
    query_embedding = model.encode([query], prompt=prompt)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )
    
    print("\nSearch Results from ChromaDB:")
    print("-" * 80)
    for doc_id, distance, content in zip(
        results['ids'][0],
        results['distances'][0],
        results['documents'][0]
    ):
        print(f"ID: {doc_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Content:\n{content}")
        print("-" * 80)

def clear_database(collection):
    results = collection.get()
    if results['ids']:
        collection.delete(ids=results['ids'])
        save_hash_record(None)
        print(f"Database cleared successfully. Removed {len(results['ids'])} documents.")
    else:
        print("Database is already empty.")

def list_passages(collection):
    results = collection.get()
    if not results['documents']:
        print("No passages found in the database.")
        return
    
    print(f"\nFound {len(results['documents'])} passages:")
    print("-" * 50)
    for i, (doc_id, doc) in enumerate(zip(results['ids'], results['documents'])):
        print(f"[{i+1}] ID: {doc_id}")
        print(f"Content: {doc[:200]}...")
        print("-" * 50)

def main():
    model = initialize_model()
    collection = setup_chromadb(model, [])
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == 'clear':
            clear_database(collection)
            return
        elif command == 'list':
            list_passages(collection)
            return
    
    passages = load_documents()
    collection = setup_chromadb(model, passages)
    
    query = sys.argv[1] if len(sys.argv) > 1 else "Who invented Hangul?"
    print(f"Using query: '{query}'")
    
    perform_search(model, collection, query)

if __name__ == "__main__":
    main()
