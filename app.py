from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from typing import Any

# LangSmith imports
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
import langsmith

# Other imports
import PyPDF2
import chromadb
from chromadb.config import Settings
import shutil
import json
import logging
from pydantic import BaseModel
import uuid

# Local imports
from career_chat_service import CareerChatService
from user_context_service import UserContextService
from ai_chatbot_service import AIChatbotService

# Load environment variables
load_dotenv()

# Configure LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
if not os.getenv("LANGSMITH_API_KEY"):
    print("Warning: LANGSMITH_API_KEY not found. LangSmith tracing will be disabled.")
    os.environ["LANGSMITH_TRACING"] = "false"

# Set project name for LangSmith
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "career-rag-system")
os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT

# Initialize LangSmith client
try:
    langsmith_client = Client()
    print(f"LangSmith tracing enabled for project: {LANGSMITH_PROJECT}")
except Exception as e:
    print(f"LangSmith initialization failed: {e}")
    langsmith_client = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serialize_datetime_objects(obj: Any) -> Any:
    """Recursively serialize datetime objects to ISO format strings"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: serialize_datetime_objects(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_datetime_objects(item) for item in obj]
    else:
        return obj

app = FastAPI(title="Career Information RAG System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
UPLOAD_DIRECTORY = "./uploaded_documents"

# Ensure directories exist
os.makedirs(CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

class TracedRetriever(Runnable):
    """Custom retriever wrapper with LangSmith tracing"""
    
    def __init__(self, retriever):
        self.retriever = retriever
    
    @langsmith.traceable(run_type="retriever", name="document_retrieval")
    def invoke(self, query: str, config: Any = None):
        """Invoke the retriever with tracing"""
        docs = self.retriever.invoke(query, config)
        
        # Log retrieval details
        logger.info(f"Retrieved {len(docs)} documents for query: {query}")
        for i, doc in enumerate(docs):
            logger.info(f"Doc {i+1}: {doc.page_content[:100]}...")
            logger.info(f"Doc {i+1} metadata: {doc.metadata}")
        
        return docs
    
    @langsmith.traceable(run_type="retriever", name="document_retrieval_async")
    async def ainvoke(self, query: str, config: Any = None):
        """Async invoke the retriever with tracing"""
        # For compatibility, most retrievers don't have async methods
        # so we'll call the sync version
        return self.invoke(query, config)

class CareerRAGSystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_API_KEY
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.vector_store = None
        self.retriever = None
        self.traced_retriever = None
        self.qa_chain = None
        self._initialize_vector_store()
        
    def _initialize_vector_store(self):
        """Initialize or load existing ChromaDB vector store"""
        try:
            self.vector_store = Chroma(
                persist_directory=CHROMA_PERSIST_DIRECTORY,
                embedding_function=self.embeddings,
                collection_name="career_documents"
            )
            
            # Check if we have existing data
            if self.vector_store._collection.count() > 0:
                logger.info(f"Loaded existing vector store with {self.vector_store._collection.count()} documents")
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                )
                self.traced_retriever = TracedRetriever(self.retriever)
                self._setup_qa_chain()
            else:
                logger.info("Empty vector store initialized")
                
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        template = """You are an expert career counselor. Use the following context to answer questions about career information.

Context: {context}

Question: {question}

Please provide a comprehensive answer focusing on:
- Career Overview
- Eligibility & Path
- Exams (if applicable)
- Free Courses (Coursera, SWAYAM, etc.)
- Best Colleges (India & Abroad)
- Job Profiles and Salaries

If the information is not available in the context, clearly state that and provide general guidance.

Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        
        # Configure LLM with callbacks for tracing
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.1
        )
        
        # Use traced retriever if available, otherwise fall back to regular retriever
        retriever_to_use = self.traced_retriever if self.traced_retriever else self.retriever
        
        self.qa_chain = (
            {"context": retriever_to_use, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    @langsmith.traceable(run_type="tool", name="process_document")
    async def process_document(self, file_path: str, file_type: str) -> List[Document]:
        """Process PDF or TXT document and extract text"""
        documents = []
        
        try:
            if file_type.lower() == 'pdf':
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            elif file_type.lower() == 'txt':
                loader = TextLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
            
        return documents

    @langsmith.traceable(run_type="tool", name="chunk_documents")
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)

    @langsmith.traceable(run_type="tool", name="add_documents_to_vector_store")
    async def add_documents_to_vector_store(self, documents: List[Document]):
        """Add documents to ChromaDB vector store"""
        try:
            # Add metadata for better retrieval
            for doc in documents:
                doc.metadata.update({
                    "timestamp": datetime.now().isoformat(),
                    "doc_type": "career_info"
                })
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Update retriever and QA chain
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            self.traced_retriever = TracedRetriever(self.retriever)
            self._setup_qa_chain()
            
            logger.info(f"Added {len(documents)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise

    @langsmith.traceable(run_type="chain", name="query_career_info")
    async def query_career_info(self, query: str, info_type: str = "general") -> str:
        """Query career information using RAG"""
        if not self.qa_chain:
            raise HTTPException(status_code=400, detail="No documents loaded. Please upload documents first.")
        
        # Enhance query based on information type
        enhanced_queries = {
            "career_overview": f"What is the career overview and general information about {query}?",
            "eligibility": f"What are the eligibility criteria and educational path for {query}?",
            "exams": f"What are the important exams and entrance tests for {query}?",
            "courses": f"What are the free courses available on Coursera, SWAYAM, and other platforms for {query}?",
            "colleges": f"What are the best colleges in India and abroad for {query}?",
            "jobs_salary": f"What are the job profiles and salary expectations for {query}?",
            "general": query
        }
        
        enhanced_query = enhanced_queries.get(info_type, query)
        
        try:
            # The retrieval will be traced automatically through the QA chain
            response = await self.qa_chain.ainvoke(enhanced_query)
            
            # Log additional context for the trace (this will be captured by the decorator)
            logger.info(f"Query processed: {query} -> {info_type}")
            logger.info(f"Response length: {len(response)} characters")
            
            return response
                
        except Exception as e:
            logger.error(f"Error querying career info: {str(e)}")
            raise

    def get_vector_store_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if not self.vector_store:
            return {"total_documents": 0, "status": "empty"}
        
        try:
            count = self.vector_store._collection.count()
            return {
                "total_documents": count,
                "status": "active" if count > 0 else "empty",
                "persist_directory": CHROMA_PERSIST_DIRECTORY
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"total_documents": 0, "status": "error", "error": str(e)}

# Pydantic models for chat flow
class ChatStartRequest(BaseModel):
    session_id: Optional[str] = None

class ChatResponseRequest(BaseModel):
    session_id: str
    user_response: str
    current_stage: str
    answers: Dict[str, str]

class ChatStatusRequest(BaseModel):
    session_id: str

# Pydantic models for AI chatbot
class ChatbotStartRequest(BaseModel):
    user_id: str

class ChatbotQueryRequest(BaseModel):
    user_id: str
    session_id: str
    question: str

class UserPreferencesRequest(BaseModel):
    session_id: str
    preferences: Dict[str, Any]

# Initialize RAG system
rag_system = CareerRAGSystem()

# Initialize chat service
chat_service = CareerChatService()

# Initialize user context service
user_context_service = UserContextService()

# Initialize AI chatbot service
ai_chatbot_service = AIChatbotService(rag_system, user_context_service)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Career Information RAG System", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = rag_system.get_vector_store_stats()
    return {
        "status": "healthy",
        "vector_store": stats,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/upload-document")
@langsmith.traceable(run_type="tool", name="upload_document_endpoint")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process PDF/TXT documents"""
    
    # Validate file type
    allowed_types = ['application/pdf', 'text/plain']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file.content_type} not supported. Only PDF and TXT files are allowed."
        )
    
    try:
        # Save uploaded file temporarily
        file_extension = file.filename.split('.')[-1].lower()
        temp_file_path = os.path.join(UPLOAD_DIRECTORY, f"temp_{datetime.now().timestamp()}.{file_extension}")
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        documents = await rag_system.process_document(temp_file_path, file_extension)
        
        # Chunk documents
        chunks = rag_system.chunk_documents(documents)
        
        # Add to vector store
        await rag_system.add_documents_to_vector_store(chunks)
        
        # Clean up temporary file
        os.remove(temp_file_path)
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "total_documents": rag_system.get_vector_store_stats()["total_documents"]
        }
        
    except Exception as e:
        # Clean up temporary file if it exists
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query-career")
@langsmith.traceable(run_type="chain", name="query_career_endpoint")
async def query_career(
    query: str = Form(...),
    info_type: str = Form(default="general")
):
    """Query career information"""
    
    valid_types = [
        "career_overview", "eligibility", "exams", 
        "courses", "colleges", "jobs_salary", "general"
    ]
    
    if info_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid info_type. Must be one of: {', '.join(valid_types)}"
        )
    
    try:
        response = await rag_system.query_career_info(query, info_type)
        return {
            "query": query,
            "info_type": info_type,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error querying career info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error querying career info: {str(e)}")

@app.get("/query-types")
async def get_query_types():
    """Get available query types"""
    return {
        "available_types": [
            {"type": "career_overview", "description": "General career information and overview"},
            {"type": "eligibility", "description": "Eligibility criteria and educational paths"},
            {"type": "exams", "description": "Important exams and entrance tests"},
            {"type": "courses", "description": "Free courses (Coursera, SWAYAM, etc.)"},
            {"type": "colleges", "description": "Best colleges in India and abroad"},
            {"type": "jobs_salary", "description": "Job profiles and salary information"},
            {"type": "general", "description": "General query (default)"}
        ]
    }

@app.get("/vector-store-stats")
async def get_vector_store_stats():
    """Get vector store statistics"""
    return rag_system.get_vector_store_stats()

@app.get("/tracing-status")
async def get_tracing_status():
    """Get LangSmith tracing status"""
    return {
        "langsmith_tracing_enabled": os.getenv("LANGSMITH_TRACING", "false").lower() == "true",
        "langsmith_api_key_set": bool(os.getenv("LANGSMITH_API_KEY")),
        "langsmith_project": os.getenv("LANGSMITH_PROJECT", "career-rag-system"),
        "langsmith_endpoint": os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        "langsmith_client_initialized": langsmith_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.delete("/clear-vector-store")
async def clear_vector_store():
    """Clear all documents from vector store"""
    try:
        # Remove the persist directory
        if os.path.exists(CHROMA_PERSIST_DIRECTORY):
            shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        
        # Reinitialize the RAG system
        global rag_system
        rag_system = CareerRAGSystem()
        
        return {"message": "Vector store cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing vector store: {str(e)}")

# Career Chat Flow Routes
@app.post("/career-chat/start")
async def start_career_chat(request: ChatStartRequest):
    """Start a new career guidance chat session"""
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Start the chat
        result = chat_service.start_chat(session_id)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error starting career chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting career chat: {str(e)}")

@app.post("/career-chat/respond")
async def respond_to_career_chat(request: ChatResponseRequest):
    """Process user response and get next question or recommendations"""
    try:
        # Process the user response
        result = chat_service.process_response(
            session_id=request.session_id,
            user_response=request.user_response,
            current_stage=request.current_stage,
            answers=request.answers
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing chat response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing chat response: {str(e)}")

@app.get("/career-chat/status/{session_id}")
async def get_chat_status(session_id: str):
    """Get current chat session status"""
    try:
        result = chat_service.get_chat_status(session_id)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error getting chat status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting chat status: {str(e)}")

@app.get("/career-chat/info")
async def get_career_chat_info():
    """Get information about the career chat flow"""
    return {
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

# AI Chatbot Routes
@app.post("/ai-chatbot/start")
@langsmith.traceable(run_type="tool", name="start_ai_chatbot")
async def start_ai_chatbot(request: ChatbotStartRequest):
    """Start a new AI chatbot session"""
    try:
        result = await ai_chatbot_service.start_chat_session(request.user_id)
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error starting AI chatbot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting AI chatbot: {str(e)}")

@app.post("/ai-chatbot/query")
@langsmith.traceable(run_type="chain", name="ai_chatbot_query_endpoint")
async def query_ai_chatbot(request: ChatbotQueryRequest):
    """Send a query to the AI chatbot"""
    try:
        result = await ai_chatbot_service.process_user_query(
            user_id=request.user_id,
            session_id=request.session_id,
            user_question=request.question
        )
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error processing AI chatbot query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing AI chatbot query: {str(e)}")

@app.get("/ai-chatbot/history/{session_id}")
async def get_chatbot_history(session_id: str):
    """Get chat history for a session"""
    try:
        messages = await ai_chatbot_service.get_session_history(session_id)
        # Serialize datetime objects
        serialized_messages = serialize_datetime_objects(messages)
        return JSONResponse(content={"session_id": session_id, "messages": serialized_messages})
        
    except Exception as e:
        logger.error(f"Error getting chatbot history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting chatbot history: {str(e)}")

@app.post("/ai-chatbot/preferences")
async def update_user_preferences(request: UserPreferencesRequest):
    """Update session preferences"""
    try:
        success = await ai_chatbot_service.update_user_preferences(
            session_id=request.session_id,
            preferences=request.preferences
        )
        
        if success:
            return JSONResponse(content={"message": "Preferences updated successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to update preferences")
            
    except Exception as e:
        logger.error(f"Error updating user preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating user preferences: {str(e)}")

@app.get("/ai-chatbot/session-context/{session_id}")
async def get_session_context(session_id: str):
    """Get session context"""
    try:
        context = user_context_service.get_user_context(session_id)
        if context:
            # Remove sensitive information
            safe_context = {
                "user_id": context.get("user_id"),
                "session_id": context.get("session_id"),
                "preferences": context.get("preferences", {}),
                "created_at": context.get("created_at"),
                "updated_at": context.get("updated_at")
            }
            # Serialize datetime objects
            serialized_context = serialize_datetime_objects(safe_context)
            return JSONResponse(content=serialized_context)
        else:
            raise HTTPException(status_code=404, detail="Session context not found")
            
    except Exception as e:
        logger.error(f"Error getting session context: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting session context: {str(e)}")

@app.get("/ai-chatbot/user-sessions/{user_id}")
async def get_user_sessions(user_id: str):
    """Get all sessions for a user"""
    try:
        contexts = user_context_service.get_user_contexts_by_user_id(user_id)
        sessions = []
        
        for context in contexts:
            # Get session details
            session = user_context_service.get_chat_session(context.get("session_id"))
            if session:
                sessions.append({
                    "session_id": context.get("session_id"),
                    "created_at": context.get("created_at"),
                    "updated_at": context.get("updated_at"),
                    "status": session.get("status"),
                    "message_count": len(session.get("messages", []))
                })
        
        # Serialize datetime objects
        serialized_sessions = serialize_datetime_objects({"sessions": sessions})
        return JSONResponse(content=serialized_sessions)
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting user sessions: {str(e)}")

@app.post("/ai-chatbot/close/{session_id}")
async def close_chatbot_session(session_id: str):
    """Close a chatbot session"""
    try:
        success = await ai_chatbot_service.close_session(session_id)
        
        if success:
            return JSONResponse(content={"message": "Session closed successfully"})
        else:
            raise HTTPException(status_code=400, detail="Failed to close session")
            
    except Exception as e:
        logger.error(f"Error closing chatbot session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error closing chatbot session: {str(e)}")

@app.get("/ai-chatbot/info")
async def get_ai_chatbot_info():
    """Get information about the AI chatbot"""
    return {
        "title": "AI Career Counselor",
        "description": "Intelligent chatbot that provides personalized career guidance using RAG and user context",
        "features": [
            "Personalized responses based on user context",
            "Integration with career knowledge base",
            "Conversation history tracking",
            "Adaptive learning from user interactions",
            "Comprehensive career guidance covering all aspects"
        ],
        "capabilities": [
            "Career exploration and recommendations",
            "Educational pathway guidance",
            "Exam and certification information",
            "Free course recommendations",
            "College and university suggestions",
            "Job market insights and salary data",
            "Skill development advice",
            "Personalized action plans"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 8000)))