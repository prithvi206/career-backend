import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from dotenv import load_dotenv
import langsmith

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from user_context_service import UserContextService

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class AIChatbotService:
    """AI-powered chatbot service with user context and RAG integration"""
    
    def __init__(self, rag_system, user_context_service: UserContextService):
        """Initialize the AI chatbot service"""
        self.rag_system = rag_system
        self.user_context_service = user_context_service
        
        # Initialize Gemini AI
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.3
        )
        
        # Setup the chatbot prompt template
        self._setup_chatbot_prompt()
        
    def _setup_chatbot_prompt(self):
        """Setup the chatbot prompt with context awareness"""
        template = """You are an expert career counselor chatbot. You have access to a comprehensive database of career information through RAG (Retrieval-Augmented Generation) and user context to provide personalized responses.

User Context:
- User ID: {user_id}
- Interests: {interests}
- Career Goals: {career_goals}
- Educational Background: {educational_background}
- Experience Level: {experience_level}
- Recent Conversation: {conversation_history}

RAG Context (if available):
{rag_context}

Current User Question: {user_question}

Instructions:
1. Use the RAG context to provide accurate, up-to-date career information
2. Personalize your response based on the user's context (interests, goals, background)
3. Be conversational and supportive
4. Focus on providing comprehensive career guidance including:
   - Career Overview
   - Eligibility & Educational Path
   - Important Exams (if applicable)
   - Free Courses (Coursera, SWAYAM, etc.)
   - Best Colleges (India & Abroad)
   - Job Profiles and Salary Expectations
   - Next Steps and Action Items
5. If information is not available in the RAG context, provide general guidance and suggest resources
6. Ask follow-up questions to better understand user needs when appropriate
7. Remember previous conversation context and build upon it

Response:"""

        self.prompt = ChatPromptTemplate.from_template(template)
        
    @langsmith.traceable(run_type="chain", name="ai_chatbot_query")
    async def process_user_query(self, user_id: str, session_id: str, user_question: str) -> Dict[str, Any]:
        """Process user query with context awareness and RAG integration"""
        try:
            # Get user context for this session
            user_context = self.user_context_service.get_user_context(session_id)
            if not user_context:
                # Create new user context if doesn't exist
                user_context = self.user_context_service.create_user_context(user_id, session_id)
            
            # Get conversation history for this session
            conversation_history = self.user_context_service.get_conversation_history(session_id, limit=5)
            
            # Get RAG context if available
            rag_context = ""
            if self.rag_system and self.rag_system.qa_chain:
                try:
                    rag_response = await self.rag_system.query_career_info(user_question, "general")
                    rag_context = rag_response
                except Exception as e:
                    logger.warning(f"RAG system unavailable: {e}")
                    rag_context = "RAG system currently unavailable. Providing general guidance based on knowledge."
            
            # Format conversation history
            formatted_history = self._format_conversation_history(conversation_history)
            
            # Prepare context for the prompt
            context_data = {
                "user_id": user_id,
                "interests": ", ".join(user_context.get("preferences", {}).get("interests", [])),
                "career_goals": ", ".join(user_context.get("preferences", {}).get("career_goals", [])),
                "educational_background": user_context.get("preferences", {}).get("educational_background", "Not specified"),
                "experience_level": user_context.get("preferences", {}).get("experience_level", "beginner"),
                "conversation_history": formatted_history,
                "rag_context": rag_context,
                "user_question": user_question
            }
            
            # Generate response using the LLM
            response = await self._generate_response(context_data)
            
            # Store the conversation
            await self._store_conversation(user_id, session_id, user_question, response)
            
            # Extract insights for context updates
            await self._update_user_context_from_query(session_id, user_question, response)
            
            return {
                "response": response,
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
                "context_used": {
                    "rag_available": bool(rag_context and "unavailable" not in rag_context),
                    "user_context_available": bool(user_context),
                    "conversation_history_length": len(conversation_history)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            raise
    
    async def _generate_response(self, context_data: Dict[str, Any]) -> str:
        """Generate AI response using the prompt template"""
        try:
            # Create the chain
            chain = self.prompt | self.llm | StrOutputParser()
            
            # Generate response
            response = await chain.ainvoke(context_data)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."
    
    async def _store_conversation(self, user_id: str, session_id: str, user_question: str, ai_response: str):
        """Store conversation in both session and user context"""
        try:
            # Store in session
            self.user_context_service.add_message_to_session(session_id, {
                "type": "user",
                "content": user_question,
                "timestamp": datetime.utcnow()
            })
            
            self.user_context_service.add_message_to_session(session_id, {
                "type": "assistant",
                "content": ai_response,
                "timestamp": datetime.utcnow()
            })
            
            # Store in session-based context
            self.user_context_service.add_to_conversation_history(session_id, {
                "type": "user",
                "content": user_question
            })
            
            self.user_context_service.add_to_conversation_history(session_id, {
                "type": "assistant",
                "content": ai_response
            })
            
        except Exception as e:
            logger.error(f"Error storing conversation: {e}")
    
    async def _update_user_context_from_query(self, session_id: str, user_question: str, ai_response: str):
        """Update session context based on the query and response"""
        try:
            # Simple keyword extraction for interests and goals
            interests_keywords = [
                "interested in", "love", "passionate about", "enjoy", "fascinated by",
                "drawn to", "excited about", "curious about"
            ]
            
            career_keywords = [
                "want to become", "aspire to be", "goal is to", "dream of being",
                "planning to be", "hoping to work as", "interested in career"
            ]
            
            # Extract potential interests
            user_context = self.user_context_service.get_user_context(session_id)
            current_interests = user_context.get("preferences", {}).get("interests", [])
            current_goals = user_context.get("preferences", {}).get("career_goals", [])
            
            # Simple extraction (in a real app, you'd use NLP)
            query_lower = user_question.lower()
            
            # Check for career-related terms
            career_terms = [
                "software engineer", "data scientist", "doctor", "teacher", "engineer",
                "designer", "manager", "developer", "analyst", "consultant", "researcher",
                "architect", "scientist", "artist", "writer", "lawyer", "nurse", "therapist"
            ]
            
            for term in career_terms:
                if term in query_lower and term not in current_interests:
                    current_interests.append(term)
            
            # Update preferences if new information found
            if current_interests != user_context.get("preferences", {}).get("interests", []):
                self.user_context_service.update_user_preferences(session_id, {
                    "interests": current_interests,
                    "career_goals": current_goals
                })
            
        except Exception as e:
            logger.error(f"Error updating user context: {e}")
    
    def _format_conversation_history(self, history: List[Dict[str, Any]]) -> str:
        """Format conversation history for the prompt"""
        if not history:
            return "No previous conversation"
        
        formatted = []
        for msg in history:
            msg_type = msg.get("type", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            
            if msg_type == "user":
                formatted.append(f"User: {content}")
            elif msg_type == "assistant":
                formatted.append(f"Assistant: {content[:200]}...")  # Truncate long responses
        
        return "\n".join(formatted[-10:])  # Last 10 messages
    
    @langsmith.traceable(run_type="tool", name="start_chat_session")
    async def start_chat_session(self, user_id: str) -> Dict[str, Any]:
        """Start a new chat session for a user"""
        try:
            # Create new chat session
            session_id = self.user_context_service.create_chat_session(user_id, "general")
            
            # Create session-based user context
            user_context = self.user_context_service.create_user_context(user_id, session_id)
            
            # Welcome message
            welcome_message = self._generate_welcome_message(user_context)
            
            # Store welcome message
            self.user_context_service.add_message_to_session(session_id, {
                "type": "assistant",
                "content": welcome_message,
                "timestamp": datetime.utcnow()
            })
            
            return {
                "session_id": session_id,
                "user_id": user_id,
                "welcome_message": welcome_message,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting chat session: {e}")
            raise
    
    def _generate_welcome_message(self, user_context: Dict[str, Any]) -> str:
        """Generate a personalized welcome message"""
        preferences = user_context.get("preferences", {})
        interests = preferences.get("interests", [])
        career_goals = preferences.get("career_goals", [])
        
        if interests or career_goals:
            interest_text = f" I see you're interested in {', '.join(interests[:3])}" if interests else ""
            goal_text = f" and have goals related to {', '.join(career_goals[:2])}" if career_goals else ""
            
            return f"""Hello! Welcome back to your career guidance chat!{interest_text}{goal_text}. 
            
I'm here to help you with any career-related questions you might have. I can provide information about:
- Career paths and opportunities
- Educational requirements and pathways
- Important exams and certifications
- Free online courses and resources
- College recommendations
- Job market insights and salary expectations
- Skill development advice

What would you like to explore today?"""
        else:
            return """Hello! Welcome to your career guidance chat! 🎯

I'm your AI career counselor, here to help you navigate your career journey. I can assist you with:

📚 **Career Exploration**: Discover career paths that match your interests
🎓 **Education Planning**: Learn about required qualifications and pathways
📝 **Exam Guidance**: Information about entrance exams and certifications
💻 **Free Resources**: Courses on Coursera, SWAYAM, and other platforms
🏫 **College Recommendations**: Best institutions in India and abroad
💼 **Job Market Insights**: Salary expectations and job profiles
🚀 **Skill Development**: What skills to focus on for your career

What career topic would you like to explore today?"""
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chat session history"""
        try:
            messages = self.user_context_service.get_session_messages(session_id)
            return messages
            
        except Exception as e:
            logger.error(f"Error getting session history: {e}")
            return []
    
    async def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """Update session preferences"""
        try:
            return self.user_context_service.update_user_preferences(session_id, preferences)
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    async def close_session(self, session_id: str) -> bool:
        """Close a chat session"""
        try:
            return self.user_context_service.close_session(session_id)
            
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return False 