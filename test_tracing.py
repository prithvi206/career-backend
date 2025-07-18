#!/usr/bin/env python3
"""
Test script to demonstrate LangSmith tracing functionality.
This script tests the RAG system with tracing enabled.
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8001"

async def test_tracing_status():
    """Test the tracing status endpoint"""
    print("🔍 Testing LangSmith tracing status...")
    
    response = requests.get(f"{BASE_URL}/tracing-status")
    if response.status_code == 200:
        status = response.json()
        print(f"✅ Tracing Status:")
        print(f"   - LangSmith Tracing: {'✅ Enabled' if status['langsmith_tracing_enabled'] else '❌ Disabled'}")
        print(f"   - API Key Set: {'✅ Yes' if status['langsmith_api_key_set'] else '❌ No'}")
        print(f"   - Project: {status['langsmith_project']}")
        print(f"   - Client Initialized: {'✅ Yes' if status['langsmith_client_initialized'] else '❌ No'}")
        return status['langsmith_tracing_enabled']
    else:
        print(f"❌ Failed to get tracing status: {response.status_code}")
        return False

async def test_health():
    """Test the health endpoint"""
    print("\n🏥 Testing health endpoint...")
    
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        health = response.json()
        print(f"✅ System Health: {health['status']}")
        print(f"   - Vector Store Documents: {health['vector_store']['total_documents']}")
        print(f"   - Vector Store Status: {health['vector_store']['status']}")
        return health['vector_store']['total_documents'] > 0
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

async def test_query_with_tracing():
    """Test querying with tracing enabled"""
    print("\n🔍 Testing career query with tracing...")
    
    # Test different query types
    test_queries = [
        {"query": "data scientist", "info_type": "career_overview"},
        {"query": "machine learning engineer", "info_type": "eligibility"},
        {"query": "software engineer", "info_type": "colleges"},
        {"query": "cybersecurity", "info_type": "general"}
    ]
    
    for test_query in test_queries:
        print(f"\n📋 Testing query: '{test_query['query']}' (type: {test_query['info_type']})")
        
        response = requests.post(
            f"{BASE_URL}/query-career",
            data=test_query
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Query successful")
            print(f"   - Response length: {len(result['response'])} characters")
            print(f"   - Timestamp: {result['timestamp']}")
            print(f"   - First 100 chars: {result['response'][:100]}...")
            
            # This query should now be traced in LangSmith
            print(f"   - 🔍 Check LangSmith dashboard for trace details")
        else:
            print(f"❌ Query failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   - Error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"   - Raw error: {response.text}")

async def test_upload_with_tracing():
    """Test document upload with tracing"""
    print("\n📄 Testing document upload with tracing...")
    
    # Check if career_data.txt exists
    career_data_path = Path("career_data.txt")
    if career_data_path.exists():
        print(f"✅ Found career_data.txt, uploading...")
        
        with open(career_data_path, "rb") as f:
            files = {"file": ("career_data.txt", f, "text/plain")}
            response = requests.post(f"{BASE_URL}/upload-document", files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Upload successful")
            print(f"   - Filename: {result['filename']}")
            print(f"   - Chunks created: {result['chunks_created']}")
            print(f"   - Total documents: {result['total_documents']}")
            print(f"   - 🔍 Check LangSmith dashboard for upload trace")
            return True
        else:
            print(f"❌ Upload failed: {response.status_code}")
            try:
                error = response.json()
                print(f"   - Error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"   - Raw error: {response.text}")
    else:
        print(f"❌ career_data.txt not found. Upload test skipped.")
    
    return False

async def main():
    """Main test function"""
    print("🚀 LangSmith Tracing Test Suite")
    print("=" * 50)
    
    # Test 1: Check tracing status
    tracing_enabled = await test_tracing_status()
    
    if not tracing_enabled:
        print("\n⚠️  WARNING: LangSmith tracing is not enabled!")
        print("   Make sure to set LANGCHAIN_API_KEY and LANGCHAIN_TRACING_V2=true")
        print("   The tests will still run but won't generate traces.")
    
    # Test 2: Health check
    has_documents = await test_health()
    
    # Test 3: Upload document (if available)
    if not has_documents:
        print("\n📋 No documents in vector store. Attempting upload...")
        await test_upload_with_tracing()
    
    # Test 4: Query with tracing
    await test_query_with_tracing()
    
    print("\n" + "=" * 50)
    print("🎉 Test suite completed!")
    
    if tracing_enabled:
        print("\n🔍 To view traces:")
        print("   1. Go to https://smith.langchain.com")
        print("   2. Navigate to your project: 'career-rag-system'")
        print("   3. View the traces generated by these tests")
        print("\nTrace types to look for:")
        print("   - upload_document_endpoint")
        print("   - query_career_endpoint")
        print("   - career_rag_query (with retrieval details)")
        print("   - process_document")
        print("   - chunk_documents")
        print("   - add_documents_to_vector_store")
    else:
        print("\n⚠️  Set up LangSmith tracing to see detailed traces!")

if __name__ == "__main__":
    asyncio.run(main()) 