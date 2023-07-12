import streamlit as st
from chat import chatsection
from streamlit_chat import message

st.title("ChatBot Interface")

import os
os.environ['CURL_CA_BUNDLE'] = '' # per https://stackoverflow.com/a/75746105

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
with st.chat_message("assistant"):
    st.write("Hi, How can I assist you?")
    
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Retrieving the data for you, please wait"):
            message_placeholder = st.empty()
            full_response = ""
            result_from_chat = chatsection(prompt)
            print(prompt, result_from_chat['answer'])
            full_response += chatsection(prompt)['answer']
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})