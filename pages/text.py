import  requests
import streamlit as st
# st.title("基于interlm的文旅小助手")


import requests
# CUDA_VISIBLE_DEVICES=0 python asr_server.py > log.txt 2>&1 &
import os


def rag_response(text:str):
    headers = {'Content-Type': 'application/json'}
   
    data = {"text": text}
    url= "http://127.0.0.1:7861/"
    response = requests.post(url+"rag1", headers=headers, json=data)
    response = response.json()
    print(response)
    if response['code'] == 0:
        res = response['res']
        return res
    else:
        return response['msg']


   
if "messages" not in st.session_state.keys():
        st.session_state.messages = []   

for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("请输入你的问题?"):
    # Display user message in chat message container
        with st.chat_message("user"):
             st.markdown(prompt)
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                with st.chat_message("assistant"):
                    stream =rag_response(prompt)
                    placeholder = st.empty()
                    placeholder.markdown(stream)
                    st.session_state.prompt =stream
            st.session_state.messages.append({"role": "assistant", "content": stream})
        print(st.session_state.messages)

