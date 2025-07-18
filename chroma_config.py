import os
import chromadb
from chromadb.config import Settings

def get_chroma_client():
    """Get ChromaDB client with proper configuration"""
    
    persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Configure ChromaDB
    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )
    
    # Create client
    client = chromadb.Client(settings)
    
    return client

def initialize_chroma_collection(client, collection_name="career_documents"):
    """Initialize or get existing collection"""
    
    try:
        # Try to get existing collection
        collection = client.get_collection(collection_name)
        print(f"✅ Loaded existing collection: {collection_name}")
    except:
        # Create new collection
        collection = client.create_collection(collection_name)
        print(f"✅ Created new collection: {collection_name}")
    
    return collection

def get_collection_stats(collection):
    """Get statistics about the collection"""
    
    try:
        count = collection.count()
        return {
            "total_documents": count,
            "status": "active" if count > 0 else "empty"
        }
    except Exception as e:
        return {
            "total_documents": 0,
            "status": "error",
            "error": str(e)
        } 