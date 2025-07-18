import pytest
import asyncio
import os
from pathlib import Path
import requests
import time

# Test configuration
BASE_URL = "http://localhost:8000"
SAMPLE_DATA_DIR = Path("sample_data")

class TestRAGSystem:
    """Complete test suite for RAG system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.base_url = BASE_URL
        
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_query_types_endpoint(self):
        """Test query types endpoint"""
        response = requests.get(f"{self.base_url}/query-types")
        assert response.status_code == 200
        data = response.json()
        assert "available_types" in data
        assert len(data["available_types"]) == 7
    
    def test_vector_store_stats(self):
        """Test vector store statistics"""
        response = requests.get(f"{self.base_url}/vector-store-stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_documents" in data
        assert "status" in data
    
    def test_document_upload(self):
        """Test document upload functionality"""
        # Create a test file
        test_file = SAMPLE_DATA_DIR / "software_engineering_career.txt"
        if test_file.exists():
            with open(test_file, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/upload-document", files=files)
                assert response.status_code == 200
                data = response.json()
                assert "message" in data
                assert "chunks_created" in data
    
    def test_career_query(self):
        """Test career query functionality"""
        # First upload a document
        self.test_document_upload()
        
        # Wait a bit for processing
        time.sleep(2)
        
        # Test query
        query_data = {
            'query': 'software engineering',
            'info_type': 'career_overview'
        }
        response = requests.post(f"{self.base_url}/query-career", data=query_data)
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "query" in data

def run_tests():
    """Run all tests"""
    print("🧪 Running RAG System Tests...")
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print("❌ Server not running. Please start the server first.")
            return
    except:
        print("❌ Cannot connect to server. Please start the server first.")
        return
    
    # Run tests
    test_suite = TestRAGSystem()
    
    tests = [
        test_suite.test_health_endpoint,
        test_suite.test_query_types_endpoint,
        test_suite.test_vector_store_stats,
        test_suite.test_document_upload,
        test_suite.test_career_query
    ]
    
    for test in tests:
        try:
            test()
            print(f"✅ {test.__name__} passed")
        except Exception as e:
            print(f"❌ {test.__name__} failed: {str(e)}")
    
    print("✅ Test suite completed!")

if __name__ == "__main__":
    run_tests() 