# streamlit_app.py
import streamlit as st
import requests
import os

# Backend URL - change this to your deployed backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.title("GenAI Chatbot")
st.markdown("Finance queries → SQL via workflow; others → RAG via fallback.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question:"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from backend
    try:
        res = requests.post(f"{BACKEND_URL}/query", data={"question": prompt})
        if res.status_code == 200:
            answer = res.json().get("answer", "")
        else:
            answer = f"Backend error: {res.text}"
    except Exception as e:
        answer = f"Connection error: {str(e)}"

    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)