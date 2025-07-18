import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from dotenv import load_dotenv
import uuid

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class UserContextService:
    """Service for managing user context in MongoDB"""
    
    def __init__(self):
        # MongoDB connection
        self.mongo_uri = os.getenv("MONGODB_URI", "mongodb+srv://admin:N7TQ0vJNl2JxNFGK@learnmongo.qmo7ywz.mongodb.net/")
        self.database_name = os.getenv("MONGODB_DATABASE", "career_chatbot")
        
        # MongoDB connection options
        self.connection_options = {
            "serverSelectionTimeoutMS": 5000,  # 5 second timeout
            "connectTimeoutMS": 10000,  # 10 second timeout
            "socketTimeoutMS": 20000,   # 20 second timeout
            "retryWrites": True,
            "w": "majority",
            "tlsAllowInvalidCertificates": True,  # For development - remove in production
            "tlsInsecure": True,  # For development - remove in production
        }
        
        try:
            self.client = MongoClient(self.mongo_uri, **self.connection_options)
            
            # Test the connection
            self.client.admin.command('ping')
            
            self.db: Database = self.client[self.database_name]
            
            # Collections
            self.user_contexts: Collection = self.db["user_contexts"]
            self.chat_sessions: Collection = self.db["chat_sessions"]
            
            # Create indexes for better performance (with error handling)
            try:
                self.user_contexts.create_index("user_id")
                self.chat_sessions.create_index("session_id")
                self.chat_sessions.create_index("user_id")
                logger.info("MongoDB indexes created successfully")
            except Exception as index_error:
                logger.warning(f"Failed to create indexes (this is non-critical): {index_error}")
            
            self.mongodb_available = True
            logger.info(f"Connected to MongoDB at {self.mongo_uri}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            logger.warning("MongoDB unavailable - using in-memory storage as fallback")
            self.mongodb_available = False
            self.client = None
            self.db = None
            self.user_contexts = None
            self.chat_sessions = None
            
            # Initialize in-memory storage
            self.memory_contexts = {}
            self.memory_sessions = {}
    
    def create_user_context(self, user_id: str, session_id: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create a new user context for a specific session"""
        try:
            context = {
                "user_id": user_id,
                "session_id": session_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "context_data": context_data or {},
                "preferences": {
                    "interests": [],
                    "career_goals": [],
                    "educational_background": "",
                    "experience_level": "beginner"
                },
                "conversation_history": [],
                "metadata": {}
            }
            
            if self.mongodb_available:
                # Insert or update user context for this session in MongoDB
                result = self.user_contexts.update_one(
                    {"session_id": session_id},
                    {"$set": context},
                    upsert=True
                )
            else:
                # Store in memory
                self.memory_contexts[session_id] = context
            
            return context
            
        except Exception as e:
            logger.error(f"Error creating user context: {e}")
            raise
    
    def get_user_context(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get user context by session ID"""
        try:
            if self.mongodb_available:
                context = self.user_contexts.find_one({"session_id": session_id})
                return context
            else:
                # Get from memory
                return self.memory_contexts.get(session_id)
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return None
    
    def get_user_contexts_by_user_id(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all contexts for a specific user"""
        try:
            if self.mongodb_available:
                contexts = list(self.user_contexts.find({"user_id": user_id}))
                return contexts
            else:
                # Get from memory
                return [ctx for ctx in self.memory_contexts.values() if ctx.get("user_id") == user_id]
            
        except Exception as e:
            logger.error(f"Error getting user contexts: {e}")
            return []
    
    def update_user_context(self, session_id: str, context_data: Dict[str, Any]) -> bool:
        """Update user context with new data"""
        try:
            update_data = {
                "updated_at": datetime.utcnow(),
                **context_data
            }
            
            if self.mongodb_available:
                result = self.user_contexts.update_one(
                    {"session_id": session_id},
                    {"$set": update_data}
                )
                return result.modified_count > 0
            else:
                # Update in memory
                if session_id in self.memory_contexts:
                    self.memory_contexts[session_id].update(update_data)
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Error updating user context: {e}")
            return False
    
    def add_to_conversation_history(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Add a message to session's conversation history"""
        try:
            message_with_timestamp = {
                **message,
                "timestamp": datetime.utcnow()
            }
            
            if self.mongodb_available:
                result = self.user_contexts.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {
                            "conversation_history": {
                                "$each": [message_with_timestamp],
                                "$slice": -50  # Keep only last 50 messages
                            }
                        },
                        "$set": {"updated_at": datetime.utcnow()}
                    }
                )
                return result.modified_count > 0
            else:
                # Update in memory
                if session_id in self.memory_contexts:
                    history = self.memory_contexts[session_id].get("conversation_history", [])
                    history.append(message_with_timestamp)
                    # Keep only last 50 messages
                    if len(history) > 50:
                        history = history[-50:]
                    self.memory_contexts[session_id]["conversation_history"] = history
                    self.memory_contexts[session_id]["updated_at"] = datetime.utcnow()
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Error adding to conversation history: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session's conversation history"""
        try:
            context = self.get_user_context(session_id)
            if context and "conversation_history" in context:
                return context["conversation_history"][-limit:]
            return []
            
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def create_chat_session(self, user_id: str, session_type: str = "general") -> str:
        """Create a new chat session"""
        try:
            session_id = str(uuid.uuid4())
            session = {
                "session_id": session_id,
                "user_id": user_id,
                "session_type": session_type,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "messages": [],
                "status": "active",
                "metadata": {}
            }
            
            if self.mongodb_available:
                self.chat_sessions.insert_one(session)
            else:
                # Store in memory
                self.memory_sessions[session_id] = session
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating chat session: {e}")
            raise
    
    def get_chat_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get chat session by session ID"""
        try:
            if self.mongodb_available:
                session = self.chat_sessions.find_one({"session_id": session_id})
                return session
            else:
                # Get from memory
                return self.memory_sessions.get(session_id)
            
        except Exception as e:
            logger.error(f"Error getting chat session: {e}")
            return None
    
    def add_message_to_session(self, session_id: str, message: Dict[str, Any]) -> bool:
        """Add a message to chat session"""
        try:
            message_with_timestamp = {
                **message,
                "timestamp": datetime.utcnow()
            }
            
            if self.mongodb_available:
                result = self.chat_sessions.update_one(
                    {"session_id": session_id},
                    {
                        "$push": {"messages": message_with_timestamp},
                        "$set": {"updated_at": datetime.utcnow()}
                    }
                )
                return result.modified_count > 0
            else:
                # Update in memory
                if session_id in self.memory_sessions:
                    self.memory_sessions[session_id]["messages"].append(message_with_timestamp)
                    self.memory_sessions[session_id]["updated_at"] = datetime.utcnow()
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Error adding message to session: {e}")
            return False
    
    def get_session_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get messages from a chat session"""
        try:
            session = self.get_chat_session(session_id)
            if session and "messages" in session:
                return session["messages"][-limit:]
            return []
            
        except Exception as e:
            logger.error(f"Error getting session messages: {e}")
            return []
    
    def update_user_preferences(self, session_id: str, preferences: Dict[str, Any]) -> bool:
        """Update session preferences"""
        try:
            if self.mongodb_available:
                result = self.user_contexts.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "preferences": preferences,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                return result.modified_count > 0
            else:
                # Update in memory
                if session_id in self.memory_contexts:
                    self.memory_contexts[session_id]["preferences"] = preferences
                    self.memory_contexts[session_id]["updated_at"] = datetime.utcnow()
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    def close_session(self, session_id: str) -> bool:
        """Close a chat session"""
        try:
            if self.mongodb_available:
                result = self.chat_sessions.update_one(
                    {"session_id": session_id},
                    {
                        "$set": {
                            "status": "closed",
                            "updated_at": datetime.utcnow()
                        }
                    }
                )
                return result.modified_count > 0
            else:
                # Update in memory
                if session_id in self.memory_sessions:
                    self.memory_sessions[session_id]["status"] = "closed"
                    self.memory_sessions[session_id]["updated_at"] = datetime.utcnow()
                    return True
                return False
            
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return False
    
    def close_connection(self):
        """Close MongoDB connection"""
        try:
            if self.mongodb_available and self.client:
                self.client.close()
                logger.info("MongoDB connection closed")
            else:
                logger.info("No MongoDB connection to close (using in-memory storage)")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}") 