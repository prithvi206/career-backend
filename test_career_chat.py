#!/usr/bin/env python3
"""
Test script for the Career Chat Flow API
"""

import requests
import json
import time
from typing import Dict, Any

# Configuration
BASE_URL = "http://localhost:8000"
CHAT_API_BASE = f"{BASE_URL}/career-chat"

def test_career_chat_flow():
    """Test the complete career chat flow"""
    print("🚀 Testing Career Chat Flow API\n")
    
    # Test 1: Get chat info
    print("1. Getting chat info...")
    try:
        response = requests.get(f"{CHAT_API_BASE}/info")
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Chat Info: {info['title']}")
            print(f"   Description: {info['description']}")
            print(f"   Total Questions: {info['total_questions']}")
            print(f"   Estimated Time: {info['estimated_time']}")
        else:
            print(f"❌ Failed to get chat info: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting chat info: {e}")
        return False
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Start chat session
    print("2. Starting chat session...")
    try:
        response = requests.post(f"{CHAT_API_BASE}/start", json={})
        if response.status_code == 200:
            chat_data = response.json()
            session_id = chat_data['session_id']
            print(f"✅ Chat started with session ID: {session_id}")
            print(f"   Message: {chat_data['message']}")
            print(f"   First Question: {chat_data['question']}")
            print(f"   Progress: {chat_data['progress']}/{chat_data['total_questions']}")
        else:
            print(f"❌ Failed to start chat: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error starting chat: {e}")
        return False
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Answer questions
    answers = {}
    current_stage = "favorite_subjects"
    
    # Sample responses for testing
    test_responses = {
        "favorite_subjects": "Math, Science, and Computer Science",
        "creativity_logic": "Logical",
        "people_tools": "Tools"
    }
    
    question_count = 1
    for stage, response in test_responses.items():
        print(f"{question_count + 2}. Answering question {question_count}...")
        print(f"   Response: {response}")
        
        try:
            payload = {
                "session_id": session_id,
                "user_response": response,
                "current_stage": stage,
                "answers": answers
            }
            
            response_req = requests.post(f"{CHAT_API_BASE}/respond", json=payload)
            
            if response_req.status_code == 200:
                chat_data = response_req.json()
                answers = chat_data.get('answers', {})
                
                if chat_data.get('complete'):
                    print(f"✅ Chat completed!")
                    print(f"   Final Message: {chat_data['message'][:200]}...")
                    print(f"   Career Options: {chat_data['career_options']}")
                    break
                else:
                    print(f"✅ Question {question_count} answered")
                    print(f"   Next Question: {chat_data.get('question', 'N/A')}")
                    print(f"   Progress: {chat_data['progress']}/{chat_data['total_questions']}")
                    current_stage = chat_data['stage']
            else:
                print(f"❌ Failed to answer question {question_count}: {response_req.status_code}")
                print(f"   Response: {response_req.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error answering question {question_count}: {e}")
            return False
        
        question_count += 1
        print("\n" + "="*50 + "\n")
    
    # Test 4: Check chat status
    print("5. Checking chat status...")
    try:
        response = requests.get(f"{CHAT_API_BASE}/status/{session_id}")
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Chat Status: {status_data['status']}")
            print(f"   Message: {status_data['message']}")
        else:
            print(f"❌ Failed to get chat status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting chat status: {e}")
        return False
    
    print("\n🎉 Career Chat Flow test completed successfully!")
    return True

def test_error_cases():
    """Test error handling"""
    print("\n" + "="*60)
    print("🔍 Testing Error Cases\n")
    
    # Test invalid session ID
    print("1. Testing invalid session ID...")
    try:
        response = requests.get(f"{CHAT_API_BASE}/status/invalid-session-id")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test malformed request
    print("\n2. Testing malformed request...")
    try:
        response = requests.post(f"{CHAT_API_BASE}/respond", json={
            "session_id": "test",
            "user_response": "test",
            # Missing current_stage and answers
        })
        print(f"   Status Code: {response.status_code}")
        if response.status_code == 422:
            print("   ✅ Correctly rejected malformed request")
        else:
            print("   ❌ Unexpected response to malformed request")
    except Exception as e:
        print(f"   Error: {e}")

def main():
    """Main test function"""
    print("Career Chat Flow API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            print(f"❌ Server not responding correctly. Status: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Server not accessible: {e}")
        print("   Please make sure the FastAPI server is running with:")
        print("   python app.py")
        return
    
    print("✅ Server is running\n")
    
    # Run tests
    success = test_career_chat_flow()
    
    if success:
        test_error_cases()
        print("\n🎉 All tests completed!")
    else:
        print("\n❌ Some tests failed!")

if __name__ == "__main__":
    main() 