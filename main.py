import streamlit as st
from model import Chatbot

# Because every time user interacts with the widgets, streamlit gonna rerun the code from
# top to bottom, we cache two functions that init chatbot and generate answer for saving memory.

@st.cache_resource
def init_chatbot():
    Chatbot()
    return Chatbot()

@st.cache_resource
def get_response(chatbot, user_input):
    return chatbot.generate_response(prompt)

chatbot = init_chatbot()

st.title("JAIST Chatbot")
with st.chat_message("assistant"):
    st.markdown("Hello, I am JAIST Chatbot. How can I help you?")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input(""):
    # prompt = st.chat_input()
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbot.generate_response(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})