import os
import logging
from typing import Dict, Any, List, Union, Optional
from dotenv import load_dotenv
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_env_vars() -> None:
    """Validate API key is available in environment."""
    api_key = None
    
    # Try getting from streamlit secrets first
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except:
        # Fall back to environment variable
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment variables.")
    logger.info("API key configured successfully")

# Load environment variables
load_dotenv()

# Modern imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field, validator
from datetime import datetime

# ------------------------------
# Pydantic Models for Structured Output
# ------------------------------

class MeetingInfo(BaseModel):
    """Model for storing basic meeting information."""
    company_name: Optional[str] = Field(default="Not Specified", description="Name of the company")
    participants: List[str] = Field(default_factory=list, description="List of meeting participants")
    location: Optional[str] = Field(default="Not Specified", description="Where the meeting took place")
    duration: Optional[str] = Field(default="Not Specified", description="How long the meeting lasted")
    employee_count: Optional[int] = Field(default=None, description="Number of employees at the company")
    management_levels: Optional[int] = Field(default=None, description="Number of management levels")
    
    @validator('participants', pre=True, always=True)
    def handle_participants(cls, v):
        if not v:
            return ["Not Specified"]
        return v if isinstance(v, list) else [str(v)]

class ActionItem(BaseModel):
    """Model for tracking action items from the meeting."""
    description: Optional[str] = Field(default="No specific action items", description="Description of the action item")
    owner: Optional[str] = Field(default="Not Assigned", description="Person responsible for the action item")
    due_date: Optional[str] = Field(default="Not Set", description="Due date or timeframe")

class MeetingMinutes(BaseModel):
    """Complete model for meeting minutes."""
    meeting_info: MeetingInfo = Field(default_factory=MeetingInfo, description="Basic meeting information")
    strategic_goals: List[str] = Field(default_factory=list, description="Company's strategic goals")
    development_focus: Optional[str] = Field(default="Not Specified", description="Company's development focus")
    challenges: List[str] = Field(default_factory=list, description="Challenges faced by target groups")
    action_items: List[ActionItem] = Field(default_factory=list, description="Action items from the meeting")
    follow_up_date: Optional[str] = Field(default="Not Set", description="When to check in on progress")
    additional_notes: Optional[str] = Field(default="No additional notes", description="Any additional relevant information")

# ------------------------------
# Utility Functions
# ------------------------------

def initialize_llm(model_name: str = "gpt-4", temperature: float = 0.7) -> ChatOpenAI:
    """Initialize the language model with API key from environment."""
    try:
        # Make sure API key is set
        validate_env_vars()
        
        return ChatOpenAI(
            model_name=model_name, 
            temperature=temperature
        )
    except Exception as e:
        logger.error(f"Error initializing ChatOpenAI: {e}")
        raise

# ------------------------------
# Chain Components
# ------------------------------

def create_structured_interview_chain(llm: ChatOpenAI):
    """Create a chain that conducts the interview and outputs structured data."""
    
    interview_system_template = """You are an AI meeting assistant that helps extract key information for meeting minutes.
You MUST respond with valid JSON that matches this exact structure:

{
    "meeting_info": {
        "company_name": "string",
        "participants": ["string"],
        "location": "string",
        "duration": "string",
        "employee_count": null or number,
        "management_levels": null or number
    },
    "strategic_goals": ["string"],
    "development_focus": "string",
    "challenges": ["string"],
    "action_items": [
        {
            "description": "string",
            "owner": "string",
            "due_date": "string"
        }
    ],
    "follow_up_date": "string",
    "additional_notes": "string"
}

If any information is missing, use "Not Specified" for strings, [] for arrays, and null for numbers.
IMPORTANT: Your response must be valid JSON and nothing else."""
    
    interview_prompt = ChatPromptTemplate.from_messages([
        ("system", interview_system_template),
        ("human", "Generate valid JSON with meeting information from this input: {input}")
    ])
    
    # Create parser for structured output
    parser = PydanticOutputParser(pydantic_object=MeetingMinutes)
    
    # Create the Interview Chain with LCEL
    return (
        interview_prompt 
        | llm 
        | parser
    ).with_config({"run_name": "Structured Interview Chain"})

def create_mom_chain(llm: ChatOpenAI):
    """Create the MoM generation chain using structured data."""
    
    mom_system_template = """You are a professional meeting assistant who creates detailed, well-formatted meeting minutes.
Using the structured data provided, create a comprehensive meeting minutes document that is ready to share with stakeholders.

Format the document using markdown for readability, with clear sections, bullet points, and highlighting of key information.
The tone should be professional and the content should be actionable."""
    
    mom_prompt = ChatPromptTemplate.from_messages([
        ("system", mom_system_template),
        ("human", "Generate professional meeting minutes from this structured data: {structured_data}")
    ])
    
    # Create the MoM Generation Chain with LCEL
    return (
        mom_prompt 
        | llm 
        | StrOutputParser()
    ).with_config({"run_name": "Enhanced MoM Generation Chain"})

def create_interactive_chain(llm: ChatOpenAI):
    """Create an interactive chat chain with memory."""
    
    system_template = """You are a helpful meeting assistant conducting an interview to gather information for meeting minutes.
Ask questions one at a time to gather all essential information for comprehensive meeting minutes.

Start with these essential questions:
1. What is the name of the company?
2. Who was present at the meeting?
3. Where did the meeting take place?
4. How long did the meeting last?
5. How many employees does the company have?
6. How many levels of management does the company have?

Then explore:
- Strategic goals
- Development focus
- Challenges
- Action items and responsibilities
- Follow-up timing

Be conversational but efficient. Ask one question at a time and wait for a response before moving to the next question."""

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    return chat_prompt | llm

# ------------------------------
# Main Process Functions
# ------------------------------

def process_meeting_data(user_responses: str) -> Dict[str, Any]:
    """Process meeting data and generate minutes."""
    try:
        llm = initialize_llm()
        
        # Create chains
        structured_chain = create_structured_interview_chain(llm)
        mom_chain = create_mom_chain(llm)
        
        try:
            # Extract structured data
            structured_data = structured_chain.invoke({"input": user_responses})
        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            # Use default values if parsing fails
            structured_data = MeetingMinutes()
        
        # Generate formatted meeting minutes
        minutes = mom_chain.invoke({"structured_data": structured_data.json()})
        
        return {
            "structured_data": structured_data,
            "meeting_minutes": minutes
        }
    except Exception as e:
        logger.error(f"Error processing meeting data: {e}")
        raise

def run_interactive_interview():
    """Run an interactive interview to collect meeting information."""
    llm = initialize_llm()
    chat_chain = create_interactive_chain(llm)
    
    chat_history = []
    print("Starting meeting minutes interview. Type 'exit' when you're finished.")
    
    # Initial greeting from the assistant
    initial_response = chat_chain.invoke({"input": "Hello", "chat_history": chat_history})
    print(f"Assistant: {initial_response.content}")
    chat_history.append(HumanMessage(content="Hello"))
    chat_history.append(initial_response)
    
    collected_responses = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        collected_responses.append(user_input)
        
        response = chat_chain.invoke({"input": user_input, "chat_history": chat_history})
        print(f"Assistant: {response.content}")
        
        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)
    
    # Compile all responses
    all_responses = "\n".join(collected_responses)
    return all_responses

if __name__ == "__main__":
    try:
        print("Welcome to the Advanced Meeting Minutes Assistant!")
        print("\n1: Enter meeting notes manually")
        print("2: Run interactive interview")
        
        choice = input("\nSelect an option (1 or 2): ")
        
        if choice == "1":
            print("\nPlease provide your meeting notes (press Ctrl+D or Ctrl+Z on Windows when finished):")
            user_input_lines = []
            
            try:
                while True:
                    line = input()
                    user_input_lines.append(line)
            except (EOFError, KeyboardInterrupt):
                pass
                
            user_responses = "\n".join(user_input_lines)
            
            # If no input provided, use demonstration data
            if not user_responses.strip():
                user_responses = """
                Company: ABC Technologies
                Participants: Sarah (CEO), Mike (CTO), Julie (HR Director), David (Team Lead)
                Location: Virtual meeting via Zoom
                Duration: 90 minutes
                Company size: 85 employees
                Management: 3 levels
                
                Strategic goals discussed:
                - Launch new product line Q3
                - Expand into European market
                - Improve employee retention by 15%
                
                Development focus: Technical upskilling of mid-level managers
                
                Challenges: 
                - Knowledge gaps in cloud architecture
                - Limited training budget
                - Rapid team growth causing communication issues
                
                Action items:
                - Julie to research training providers by next Friday
                - Mike to outline skill requirements by Wednesday
                - David to gather feedback from team by Monday
                - Follow-up meeting scheduled for two weeks from now
                """
        else:
            user_responses = run_interactive_interview()
            
        # Process the data
        results = process_meeting_data(user_responses)
        
        # Display results
        print("\n--- Structured Meeting Data ---\n")
        print(results["structured_data"])
        
        print("\n--- Generated Meeting Minutes ---\n")
        print(results["meeting_minutes"])
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")
        print("Please ensure your OpenAI API key is correctly set in the .env file.")
