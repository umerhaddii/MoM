import streamlit as st
from app import process_meeting_data, initialize_llm, create_interactive_chain
from langchain_core.messages import HumanMessage, AIMessage

# Custom CSS to position input at bottom
st.markdown("""
    <style>
        .stChatFloatingInputContainer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 1rem;
            background-color: white;
            z-index: 100;
        }
        [data-testid="stChatMessageContainer"] {
            padding-bottom: 100px;
        }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": """Welcome to the Advanced Meeting Minutes Assistant!

1: Enter meeting notes manually
2: Run interactive interview

Please select an option (1 or 2)"""
            }
        ]
        st.session_state.waiting_for_option = True  # Add this flag
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "collected_responses" not in st.session_state:
        st.session_state.collected_responses = []

def main():
    st.title("Meeting Minutes Assistant")
    init_session_state()

    # Initialize chat components
    llm = initialize_llm()
    chat_chain = create_interactive_chain(llm)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Generate minutes button (only show after some messages)
    if len(st.session_state.messages) > 3:
        if st.button("Generate Meeting Minutes", type="primary"):
            all_responses = "\n".join(st.session_state.collected_responses)
            with st.spinner("Generating meeting minutes..."):
                results = process_meeting_data(all_responses)
                st.markdown(results["meeting_minutes"])

    # Chat input
    if user_input := st.chat_input("Type your response..."):
        # Handle option selection
        if st.session_state.get("waiting_for_option", False):
            if user_input in ["1", "2"]:
                st.session_state.waiting_for_option = False
                if user_input == "1":
                    next_message = "Please enter your meeting notes directly:"
                else:
                    next_message = """Let's start with some basic information:
 What is the name of the company?"""
                
                # Add user choice and bot's next message
                st.session_state.messages.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": next_message}
                ])
                st.session_state.collected_responses.append(user_input)
                st.rerun()
            else:
                # Invalid option selected
                st.session_state.messages.extend([
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": "Please select either 1 or 2:"}
                ])
                st.rerun()
        else:
            # Normal chat flow
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.collected_responses.append(user_input)

            # Get AI response
            response = chat_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.chat_history
            })

            # Add AI response
            st.session_state.messages.append({"role": "assistant", "content": response.content})
            st.session_state.chat_history.extend([
                HumanMessage(content=user_input),
                response
            ])
            
            st.rerun()

if __name__ == "__main__":
    main()
