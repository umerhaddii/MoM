import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import streamlit as st

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_env_vars() -> None:
    """Validate API key is available in environment."""
    api_key = None
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
        os.environ["OPENAI_API_KEY"] = api_key
    except Exception as e:
        logger.warning("Falling back to environment variable for API key.")
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment variables.")
    logger.info("API key configured successfully")

# Load environment variables
load_dotenv()

# Modern imports from LangChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from datetime import datetime

# ------------------------------
# Utility Functions
# ------------------------------

def initialize_llm(model_name: str = "gpt-4", temperature: float = 0.7) -> ChatOpenAI:
    """Initialize the language model with API key from environment."""
    try:
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

def create_raw_interview_chain(llm: ChatOpenAI):
    """
    Create an interview chain that collects raw meeting responses.
    The system prompt is designed to ask essential and dynamic optional questions.
    """
    interview_system_template = (
        "You are a highly intelligent meeting assistant designed to gather all necessary information for creating accurate and detailed Meeting Minutes (MoM). "
        "Your task is to conduct an interview with a consultant by asking questions in a natural, conversational manner. Begin by asking the essential questions below:\n\n"
        "*Essential Questions:*\n"
        "1. What is the name of the company?\n"
        "2. Who was present at the meeting?\n"
        "3. Where did the meeting take place?\n"
        "4. How long did the meeting last?\n"
        "5. How many employees does the company have?\n"
        "6. How many levels of management does the company have?\n\n"
        "After receiving responses to these, analyze the context of the conversation to determine if additional details are needed. Based on the context, ask any relevant optional questions from the following list to enrich the meeting record:\n\n"
        "*Optional Questions (ask if context indicates relevance):*\n"
        "- What are the company’s main strategic goals for this period?\n"
        "- What is the company’s focus when it comes to development?\n"
        "- Which target groups within the company are prioritized for development?\n"
        "- What are the main challenges these target groups are currently facing?\n"
        "- Are there any specific competencies or skills the company wants to prioritize across teams?\n"
        "- What learning and development programs are currently in place?\n"
        "- How do you currently measure skill levels and identify training needs?\n"
        "- Which learning formats do employees prefer (online programs, in-person workshops, blended learning)?\n"
        "- Is there any specific format desired for the development (trainings, training days, team building, coaching, etc.)?\n\n"
        "Once you have gathered all the necessary and contextually relevant responses, also ask:\n"
        "- What are the key action items from this discussion?\n"
        "- Who is responsible for following up on these topics?\n"
        "- When should we check in again on the progress of development initiatives?\n\n"
        "Before concluding, confirm with the user if there is any additional information they would like to add. "
        "Your goal is to ensure that every piece of relevant data is captured in a complete meeting record. "
        "Always maintain a conversational tone, adapt your questions based on previous responses, and guide the conversation naturally."
    )
    
    interview_prompt = ChatPromptTemplate.from_messages([
        ("system", interview_system_template),
        ("human", "{input}")
    ])
    
    # We use StrOutputParser to pass through the raw text responses
    return interview_prompt | llm | StrOutputParser()

def create_mom_chain(llm: ChatOpenAI):
    """
    Create the Meeting Minutes (MoM) generation chain.
    This chain uses a high-IQ prompt to generate a professional MoM in your desired structure.
    """
    mom_system_template = (
        "You are a highly professional meeting assistant and expert consultant. Based on the raw meeting data provided, "
        "generate the final Meeting Minutes (MoM) using the following format exactly:\n\n"
        "Date: [Insert Date, e.g., 'October 15, 2024']\n"
        "Attendees:\n"
        "- Atria: [Insert Names, e.g., 'John Doe, Jane Smith']\n"
        "- Company: [Insert Names and Company, e.g., 'Alice Brown, Bob Green - XYZ Corp']\n"
        "Duration: [Insert Duration, e.g., '45 minutes']\n"
        "Location: [Insert Location, e.g., 'Atria Office' or 'Client’s Office']\n\n"
        "Company Details (if provided):\n"
        "- Number of employees: [Insert Number, e.g., '50']\n"
        "- Levels of management: [Insert Number, e.g., '3']\n\n"
        "Meeting Topic: [Insert Topic, e.g., 'Discussing training needs for leadership development']\n\n"
        "#### Key Discussion Points:\n"
        "- Client’s Current Situation and Needs: [Summarize client challenges or goals, e.g., 'Need training for 10 managers']\n"
        "- Proposed Solutions or Programs: [List ideas discussed, e.g., 'Suggested leadership workshop']\n"
        "- Agreements or Decisions: [Note outcomes, e.g., 'Agreed to propose a 2-day program']\n\n"
        "#### Action Items:\n"
        "- Next Steps: [List tasks, e.g., 'Draft proposal for leadership training']\n"
        "- Responsible Persons: [Assign names, e.g., 'John Doe']\n"
        "- Deadlines or Follow-up Dates: [Insert dates, e.g., 'October 20, 2024']\n\n"
        "Use the raw meeting data provided to fill in these placeholders. If certain optional information is missing, omit that section or mark it as 'Not Specified.' "
        "Your output must be professional, concise, and ready for stakeholder review."
    )
    
    mom_prompt = ChatPromptTemplate.from_messages([
        ("system", mom_system_template),
        ("human", "Generate the final Meeting Minutes (MoM) from the following raw meeting data:\n{raw_data}")
    ])
    
    return mom_prompt | llm | StrOutputParser()

def create_interactive_chain(llm: ChatOpenAI):
    """Create an interactive chat chain for a conversational interview."""
    system_template = (
        "You are a helpful meeting assistant conducting an interview to gather information for meeting minutes. "
        "Ask questions one at a time to gather all essential details. Begin with the following essential questions:\n"
        "1. What is the name of the company?\n"
        "2. Who was present at the meeting?\n"
        "3. Where did the meeting take place?\n"
        "4. How long did the meeting last?\n"
        "5. How many employees does the company have?\n"
        "6. How many levels of management does the company have?\n"
        "Then, ask additional questions about strategic goals, development focus, challenges, action items, and follow-up. "
        "Be conversational and wait for each response before proceeding."
    )
    
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
    """
    Process meeting data by using the raw interview chain to gather responses,
    then passing that raw data to the MoM generation chain.
    """
    try:
        llm = initialize_llm()
        interview_chain = create_raw_interview_chain(llm)
        mom_chain = create_mom_chain(llm)
        
        # Get raw meeting data (as free-form text)
        raw_data = interview_chain.invoke({"input": user_responses})
        logger.info("Raw Interview Data Collected:")
        logger.info(raw_data)
        
        # Generate the final MoM using the raw data
        minutes = mom_chain.invoke({"raw_data": raw_data})
        logger.info("Final Meeting Minutes Generated.")
        
        return {
            "raw_data": raw_data,
            "meeting_minutes": minutes
        }
    except Exception as e:
        logger.error(f"Error processing meeting data: {e}")
        raise

def run_interactive_interview() -> str:
    """
    Run an interactive interview to collect meeting information using conversation history.
    """
    llm = initialize_llm()
    chat_chain = create_interactive_chain(llm)
    
    chat_history = []
    print("Starting meeting minutes interview. Type 'exit' when you're finished.")
    
    # Initial greeting
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
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(response)
    
    # Combine all responses into a single raw data string
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
            if not user_responses.strip():
                user_responses = (
                    "Company: ABC Technologies\n"
                    "Participants: Sarah (CEO), Mike (CTO), Julie (HR Director), David (Team Lead)\n"
                    "Location: Virtual meeting via Zoom\n"
                    "Duration: 90 minutes\n"
                    "Company size: 85 employees\n"
                    "Management: 3 levels\n\n"
                    "Strategic goals discussed:\n"
                    "- Launch new product line Q3\n"
                    "- Expand into European market\n"
                    "- Improve employee retention by 15%\n\n"
                    "Development focus: Technical upskilling of mid-level managers\n\n"
                    "Challenges:\n"
                    "- Knowledge gaps in cloud architecture\n"
                    "- Limited training budget\n"
                    "- Rapid team growth causing communication issues\n\n"
                    "Action items:\n"
                    "- Julie to research training providers by next Friday\n"
                    "- Mike to outline skill requirements by Wednesday\n"
                    "- David to gather feedback from team by Monday\n"
                    "- Follow-up meeting scheduled for two weeks from now\n"
                )
        else:
            user_responses = run_interactive_interview()
        
        results = process_meeting_data(user_responses)
        
        print("\n--- Raw Interview Data ---\n")
        print(results["raw_data"])
        
        print("\n--- Generated Meeting Minutes ---\n")
        print(results["meeting_minutes"])
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        print(f"Error: {e}")
        print("Please ensure your OpenAI API key is correctly set in the .env file.")
