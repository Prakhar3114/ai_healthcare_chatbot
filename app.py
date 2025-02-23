import streamlit as st
import nltk
from transformers import pipeline
from nltk.tokenize import word_tokenize

# Download required NLTK packages
nltk.download('punkt')

# Load GPT-2 Model
chatbot = pipeline("text-generation", model="distilgpt2")


# Define a function to analyze user input and provide appropriate responses
def healthcare_chatbot(user_input):
    user_input = user_input.lower()

    # Basic healthcare responses
    if any(word in user_input for word in ["symptoms", "feeling sick", "not well"]):
        return "🩺 It’s best to consult a doctor for an accurate diagnosis. Let me know if you need help finding one."

    elif any(word in user_input for word in ["appointment", "schedule", "book"]):
        return "📅 Would you like to schedule an appointment with a doctor? I can assist you with that!"

    elif any(word in user_input for word in ["medicine", "medication", "prescription"]):
        return "💊 It's important to take prescribed medicines regularly. If you have any concerns, consult your doctor."

    # Generate AI-based responses for general queries
    else:
        response = chatbot(user_input, max_length=500, num_return_sequences=1)
        return response[0]['generated_text']


# Streamlit UI Setup
def main():
    st.set_page_config(page_title="Healthcare Chatbot", page_icon="🩺", layout="centered")

    # Title and UI Design
    st.title("🤖 AI Healthcare Assistant")
    st.markdown("### Hello! I’m here to help with your health-related queries. Ask me anything!")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    user_input = st.chat_input("Type your question here...")

    if user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
                st.markdown(response)

        # Append AI response
        st.session_state.messages.append({"role": "assistant", "content": response})


# Run the Streamlit app
if __name__ == "__main__":
    main()

