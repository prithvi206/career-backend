import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from enum import Enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import langsmith

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class ChatStage(str, Enum):
    WELCOME = "welcome"
    FAVORITE_SUBJECTS = "favorite_subjects"
    CREATIVITY_LOGIC = "creativity_logic"
    PEOPLE_TOOLS = "people_tools"
    RECOMMENDATION = "recommendation"
    COMPLETE = "complete"

class CareerChatService:
    def __init__(self):
        """Initialize the career chat service with Gemini AI"""
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.gemini_api_key,
            temperature=0.3
        )
        
        # Chat questions and flow
        self.questions = {
            ChatStage.FAVORITE_SUBJECTS: {
                "question": "What are your favorite subjects in school? (e.g., Math, Science, English, History, Arts, etc.)",
                "context": "Understanding academic interests helps identify suitable career paths."
            },
            ChatStage.CREATIVITY_LOGIC: {
                "question": "Do you prefer creativity-based work or logic-based work? (Creative/Logical/Both)",
                "context": "This helps determine if you're more suited for artistic/creative fields or analytical/technical fields."
            },
            ChatStage.PEOPLE_TOOLS: {
                "question": "Do you prefer working with people or working with tools/technology? (People/Tools/Both)",
                "context": "This indicates whether you're more suited for people-oriented careers or technical/hands-on careers."
            }
        }
        
        # Career mapping database
        self.career_options = {
            "STEM_LOGICAL_TOOLS": [
                "Software Engineer",
                "Data Scientist",
                "Mechanical Engineer",
                "Electrical Engineer",
                "Research Scientist"
            ],
            "STEM_LOGICAL_PEOPLE": [
                "Science Teacher",
                "Medical Doctor",
                "Engineering Manager",
                "Technical Consultant",
                "Research Team Leader"
            ],
            "ARTS_CREATIVE_TOOLS": [
                "Graphic Designer",
                "Web Designer",
                "Photographer",
                "Video Editor",
                "Digital Artist"
            ],
            "ARTS_CREATIVE_PEOPLE": [
                "Art Teacher",
                "Marketing Manager",
                "Content Creator",
                "Event Planner",
                "Creative Director"
            ],
            "BUSINESS_LOGICAL_PEOPLE": [
                "Business Analyst",
                "Management Consultant",
                "Project Manager",
                "Financial Advisor",
                "Operations Manager"
            ],
            "BUSINESS_LOGICAL_TOOLS": [
                "Data Analyst",
                "Financial Analyst",
                "Market Research Analyst",
                "Systems Analyst",
                "Quality Assurance Manager"
            ],
            "GENERAL_PEOPLE": [
                "Teacher",
                "Social Worker",
                "Counselor",
                "Human Resources Manager",
                "Customer Service Manager"
            ],
            "GENERAL_TOOLS": [
                "Technician",
                "Maintenance Engineer",
                "Laboratory Assistant",
                "IT Support Specialist",
                "Technical Writer"
            ]
        }

    def start_chat(self, session_id: str) -> Dict[str, Any]:
        """Start a new chat session"""
        return {
            "session_id": session_id,
            "stage": ChatStage.WELCOME,
            "message": "Welcome to the Career Guidance Chat! I'll ask you a few questions to help find suitable career options for you. Let's begin!",
            "question": self.questions[ChatStage.FAVORITE_SUBJECTS]["question"],
            "progress": 0,
            "total_questions": len(self.questions),
            "answers": {}
        }

    def process_response(self, session_id: str, user_response: str, current_stage: str, answers: Dict[str, str]) -> Dict[str, Any]:
        """Process user response and move to next stage"""
        try:
            stage = ChatStage(current_stage)
            
            # Store the answer
            answers[stage] = user_response
            
            # Determine next stage
            if stage == ChatStage.FAVORITE_SUBJECTS:
                next_stage = ChatStage.CREATIVITY_LOGIC
            elif stage == ChatStage.CREATIVITY_LOGIC:
                next_stage = ChatStage.PEOPLE_TOOLS
            elif stage == ChatStage.PEOPLE_TOOLS:
                next_stage = ChatStage.RECOMMENDATION
            else:
                next_stage = ChatStage.COMPLETE
            
            if next_stage == ChatStage.RECOMMENDATION:
                # Generate career recommendations
                recommendations = self._generate_recommendations(answers)
                return {
                    "session_id": session_id,
                    "stage": next_stage,
                    "message": recommendations["message"],
                    "career_options": recommendations["career_options"],
                    "progress": len(self.questions),
                    "total_questions": len(self.questions),
                    "answers": answers,
                    "complete": True
                }
            else:
                # Move to next question
                return {
                    "session_id": session_id,
                    "stage": next_stage,
                    "message": f"Thank you for your response! Next question:",
                    "question": self.questions[next_stage]["question"],
                    "progress": len([s for s in self.questions.keys() if s.value <= stage.value]),
                    "total_questions": len(self.questions),
                    "answers": answers
                }
                
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            return {
                "session_id": session_id,
                "stage": current_stage,
                "message": "I'm sorry, there was an error processing your response. Please try again.",
                "error": str(e)
            }

    @langsmith.traceable(run_type="chain", name="career_recommendation_generation")
    def _generate_recommendations(self, answers: Dict[str, str]) -> Dict[str, Any]:
        """Generate career recommendations based on user answers"""
        try:
            # Extract user preferences
            subjects = answers.get(ChatStage.FAVORITE_SUBJECTS, "").lower()
            creativity_logic = answers.get(ChatStage.CREATIVITY_LOGIC, "").lower()
            people_tools = answers.get(ChatStage.PEOPLE_TOOLS, "").lower()
            
            # Determine career category
            career_category = self._determine_career_category(subjects, creativity_logic, people_tools)
            
            # Get career options
            career_options = self.career_options.get(career_category, self.career_options["GENERAL_PEOPLE"])
            
            # Use Gemini AI to generate personalized recommendations
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content="""You are a career counselor helping students find suitable career paths. 
                Based on the user's responses, provide personalized career recommendations with explanations."""),
                HumanMessage(content=f"""
                Based on these responses:
                - Favorite subjects: {answers.get(ChatStage.FAVORITE_SUBJECTS, '')}
                - Preference: {answers.get(ChatStage.CREATIVITY_LOGIC, '')}
                - Work style: {answers.get(ChatStage.PEOPLE_TOOLS, '')}
                
                Here are some career options that match their profile: {', '.join(career_options[:3])}
                
                Please provide:
                1. A brief personalized message explaining why these careers suit them
                2. For each of the top 3 career options, provide:
                   - Career title
                   - Why it matches their interests
                   - Key skills needed
                   - Potential growth opportunities
                
                Keep the response encouraging and informative, suitable for a student.
                """)
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({})
            
            return {
                "message": response.content,
                "career_options": career_options[:3],
                "reasoning": {
                    "subjects": subjects,
                    "creativity_logic": creativity_logic,
                    "people_tools": people_tools,
                    "category": career_category
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return {
                "message": "Based on your responses, here are some career options that might interest you. Consider exploring these fields further to find what truly excites you!",
                "career_options": ["General Career Path 1", "General Career Path 2", "General Career Path 3"],
                "error": str(e)
            }

    def _determine_career_category(self, subjects: str, creativity_logic: str, people_tools: str) -> str:
        """Determine career category based on user responses"""
        
        # Check for STEM subjects
        stem_keywords = ["math", "science", "physics", "chemistry", "biology", "computer", "engineering", "technology"]
        is_stem = any(keyword in subjects for keyword in stem_keywords)
        
        # Check for arts/creative subjects
        arts_keywords = ["art", "english", "literature", "music", "drama", "creative", "design", "history"]
        is_arts = any(keyword in subjects for keyword in arts_keywords)
        
        # Check for business subjects
        business_keywords = ["economics", "business", "commerce", "accounting", "finance", "management"]
        is_business = any(keyword in subjects for keyword in business_keywords)
        
        # Determine preferences
        prefers_creative = "creative" in creativity_logic or "both" in creativity_logic
        prefers_logical = "logical" in creativity_logic or "logic" in creativity_logic or "both" in creativity_logic
        prefers_people = "people" in people_tools or "both" in people_tools
        prefers_tools = "tools" in people_tools or "technology" in people_tools or "both" in people_tools
        
        # Determine category
        if is_stem and prefers_logical and prefers_tools:
            return "STEM_LOGICAL_TOOLS"
        elif is_stem and prefers_logical and prefers_people:
            return "STEM_LOGICAL_PEOPLE"
        elif is_arts and prefers_creative and prefers_tools:
            return "ARTS_CREATIVE_TOOLS"
        elif is_arts and prefers_creative and prefers_people:
            return "ARTS_CREATIVE_PEOPLE"
        elif is_business and prefers_logical and prefers_people:
            return "BUSINESS_LOGICAL_PEOPLE"
        elif is_business and prefers_logical and prefers_tools:
            return "BUSINESS_LOGICAL_TOOLS"
        elif prefers_people:
            return "GENERAL_PEOPLE"
        elif prefers_tools:
            return "GENERAL_TOOLS"
        else:
            return "GENERAL_PEOPLE"

    def get_chat_status(self, session_id: str) -> Dict[str, Any]:
        """Get current chat status (placeholder for session management)"""
        return {
            "session_id": session_id,
            "status": "active",
            "message": "Chat session is active"
        } 