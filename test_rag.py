import requests
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def upload_document(file_path):
    """Test document upload"""
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post(f"{BASE_URL}/upload-document", files=files)
        print("Upload Response:", response.json())

def query_career(query, info_type="general"):
    """Test career query"""
    data = {
        'query': query,
        'info_type': info_type
    }
    response = requests.post(f"{BASE_URL}/query-career", data=data)
    print(f"Query Response ({info_type}):", response.json())

def get_query_types():
    """Get available query types"""
    response = requests.get(f"{BASE_URL}/query-types")
    print("Available Query Types:", response.json())

def get_stats():
    """Get vector store stats"""
    response = requests.get(f"{BASE_URL}/vector-store-stats")
    print("Vector Store Stats:", response.json())

if __name__ == "__main__":
    # Test the RAG system
    print("Testing Career RAG System...")
    
    # Health check
    test_health()
    
    # Get query types
    get_query_types()
    
    # Get initial stats
    get_stats()
    
    # Example: Upload a document (you need to have a PDF/TXT file)
    # upload_document("path/to/your/career_document.pdf")
    
    # Example queries
    queries = [
        ("software engineering", "career_overview"),
        ("data science", "eligibility"),
        ("machine learning", "courses"),
        ("computer science", "colleges"),
        ("AI engineer", "jobs_salary")
    ]
    
    for query, info_type in queries:
        print(f"\n--- Testing: {query} ({info_type}) ---")
        query_career(query, info_type) 