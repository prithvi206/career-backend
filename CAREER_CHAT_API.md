# Career Chat Flow API Documentation

## Overview

The Career Chat Flow API provides an interactive guided conversation to help students discover suitable career paths based on their interests and preferences. It uses Google's Gemini AI to generate personalized career recommendations.

## Features

- **Interactive Chat Flow**: Guided questions to understand user preferences
- **Personalized Recommendations**: AI-generated career suggestions based on responses
- **Session Management**: Maintains conversation state across multiple requests
- **Progress Tracking**: Shows completion progress throughout the chat
- **Error Handling**: Robust error handling and validation

## API Endpoints

### 1. Get Chat Information

```
GET /career-chat/info
```

Returns information about the career chat flow including questions and estimated time.

**Response:**

```json
{
  "title": "Career Guidance Chat",
  "description": "Interactive chat flow to help students discover suitable career paths",
  "questions": [
    "What are your favorite subjects in school?",
    "Do you prefer creativity-based work or logic-based work?",
    "Do you prefer working with people or working with tools/technology?"
  ],
  "total_questions": 3,
  "estimated_time": "2-3 minutes"
}
```

### 2. Start Chat Session

```
POST /career-chat/start
```

Starts a new career guidance chat session.

**Request Body:**

```json
{
  "session_id": "optional-custom-session-id"
}
```

**Response:**

```json
{
  "session_id": "generated-uuid-or-custom-id",
  "stage": "welcome",
  "message": "Welcome to the Career Guidance Chat! I'll ask you a few questions to help find suitable career options for you. Let's begin!",
  "question": "What are your favorite subjects in school? (e.g., Math, Science, English, History, Arts, etc.)",
  "progress": 0,
  "total_questions": 3,
  "answers": {}
}
```

### 3. Respond to Questions

```
POST /career-chat/respond
```

Process user response and get next question or final recommendations.

**Request Body:**

```json
{
  "session_id": "your-session-id",
  "user_response": "Math, Science, and Computer Science",
  "current_stage": "favorite_subjects",
  "answers": {}
}
```

**Response (Next Question):**

```json
{
  "session_id": "your-session-id",
  "stage": "creativity_logic",
  "message": "Thank you for your response! Next question:",
  "question": "Do you prefer creativity-based work or logic-based work? (Creative/Logical/Both)",
  "progress": 1,
  "total_questions": 3,
  "answers": {
    "favorite_subjects": "Math, Science, and Computer Science"
  }
}
```

**Response (Final Recommendations):**

```json
{
  "session_id": "your-session-id",
  "stage": "recommendation",
  "message": "Based on your love for Math, Science, and Computer Science, combined with your preference for logical thinking and working with tools/technology, here are some excellent career paths that align with your interests:\n\n1. **Software Engineer**\n   - Why it matches: Your strong foundation in math and computer science, plus logical thinking skills\n   - Key skills needed: Programming languages, problem-solving, algorithmic thinking\n   - Growth opportunities: Senior developer, tech lead, software architect\n\n2. **Data Scientist**\n   - Why it matches: Combines your love for math and science with analytical thinking\n   - Key skills needed: Statistics, machine learning, programming (Python/R)\n   - Growth opportunities: Senior data scientist, ML engineer, data science manager\n\n3. **Mechanical Engineer**\n   - Why it matches: Strong math and science background with hands-on technical work\n   - Key skills needed: CAD software, physics principles, problem-solving\n   - Growth opportunities: Senior engineer, project manager, engineering consultant",
  "career_options": [
    "Software Engineer",
    "Data Scientist",
    "Mechanical Engineer"
  ],
  "progress": 3,
  "total_questions": 3,
  "answers": {
    "favorite_subjects": "Math, Science, and Computer Science",
    "creativity_logic": "Logical",
    "people_tools": "Tools"
  },
  "complete": true
}
```

### 4. Check Chat Status

```
GET /career-chat/status/{session_id}
```

Get current status of a chat session.

**Response:**

```json
{
  "session_id": "your-session-id",
  "status": "active",
  "message": "Chat session is active"
}
```

## Chat Flow Stages

The chat flow progresses through these stages:

1. **`welcome`** - Initial greeting and first question
2. **`favorite_subjects`** - Ask about academic interests
3. **`creativity_logic`** - Ask about work style preference
4. **`people_tools`** - Ask about collaboration preference
5. **`recommendation`** - Generate and return career recommendations
6. **`complete`** - Chat session completed

## Career Categories

The system categorizes users into career paths based on their responses:

- **STEM + Logical + Tools**: Software Engineer, Data Scientist, Mechanical Engineer
- **STEM + Logical + People**: Science Teacher, Medical Doctor, Engineering Manager
- **Arts + Creative + Tools**: Graphic Designer, Web Designer, Photographer
- **Arts + Creative + People**: Art Teacher, Marketing Manager, Content Creator
- **Business + Logical + People**: Business Analyst, Management Consultant, Project Manager
- **Business + Logical + Tools**: Data Analyst, Financial Analyst, Systems Analyst
- **General + People**: Teacher, Social Worker, Counselor
- **General + Tools**: Technician, IT Support, Technical Writer

## Example Usage

### Complete Chat Flow Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Start chat
response = requests.post(f"{BASE_URL}/career-chat/start", json={})
chat_data = response.json()
session_id = chat_data['session_id']

# 2. Answer first question
response = requests.post(f"{BASE_URL}/career-chat/respond", json={
    "session_id": session_id,
    "user_response": "Math, Science, and Computer Science",
    "current_stage": "favorite_subjects",
    "answers": {}
})
chat_data = response.json()

# 3. Answer second question
response = requests.post(f"{BASE_URL}/career-chat/respond", json={
    "session_id": session_id,
    "user_response": "Logical",
    "current_stage": "creativity_logic",
    "answers": chat_data['answers']
})
chat_data = response.json()

# 4. Answer third question (final)
response = requests.post(f"{BASE_URL}/career-chat/respond", json={
    "session_id": session_id,
    "user_response": "Tools",
    "current_stage": "people_tools",
    "answers": chat_data['answers']
})
recommendations = response.json()

print("Career Recommendations:")
print(recommendations['message'])
print("Career Options:", recommendations['career_options'])
```

## Setup and Installation

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables:**

   ```bash
   export GEMINI_API_KEY="your-gemini-api-key"
   export LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional
   ```

3. **Run the Server:**

   ```bash
   python app.py
   ```

4. **Test the API:**
   ```bash
   python test_career_chat.py
   ```

## Error Handling

The API provides comprehensive error handling:

- **400 Bad Request**: Invalid request format
- **422 Unprocessable Entity**: Missing required fields
- **500 Internal Server Error**: Server-side errors (logged for debugging)

## Technologies Used

- **FastAPI**: Web framework for building APIs
- **LangChain**: Framework for LLM applications
- **Google Gemini AI**: Language model for generating recommendations
- **LangSmith**: Tracing and monitoring (optional)
- **Pydantic**: Data validation and serialization

## Security Considerations

- Session IDs are generated using UUID4 for security
- API keys are loaded from environment variables
- Input validation using Pydantic models
- Comprehensive error handling without exposing sensitive information

## Future Enhancements

Potential improvements for the career chat flow:

1. **Session Persistence**: Store chat sessions in database
2. **Advanced Categorization**: More sophisticated career matching algorithm
3. **Integration with Job Market Data**: Include market trends and salary information
4. **Multi-language Support**: Support for different languages
5. **Enhanced Recommendations**: Include educational pathways and skill development plans
6. **Analytics Dashboard**: Track user interactions and popular career paths
