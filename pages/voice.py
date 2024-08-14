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



def tts_api(text:str):
    headers = {'Content-Type': 'application/json'}
    print(text)
    data = {"text": text}
    url = "http://127.0.0.1:7863/"
    response = requests.post(url+"tts", headers=headers, json=data)
    print(response)
    response = response.json()
    print(response)
    if response['code'] == 0:
        res = response['res']
        return res
    else:
        return response['msg']


    # st.header("列-1")
if "voice" not in st.session_state.keys():
        st.session_state.voice = []   

for voice in st.session_state.voice:
        with st.chat_message(voice["role"]):
            if voice["role"] == "user":
                 st.markdown(voice["content"])
            else:
                path=voice["content"]
                audio_file = open(path, 'rb')
                audio_bytes = audio_file.read()
                    # 使用st.audio函数播放音频
                st.audio(audio_bytes, format='audio/wav')
if prompt := st.chat_input("请输入你的问题?"):
    # Display user message in chat message container
        with st.chat_message("user"):
             st.markdown(prompt)
    # Add user message to chat history
        st.session_state.voice.append({"role": "user", "content": prompt})

        if st.session_state.voice[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                with st.chat_message("assistant"):
                    stream =rag_response(prompt)
                    path=tts_api(stream)
                    audio_file = open(path, 'rb')
                    audio_bytes = audio_file.read()
                    # 使用st.audio函数播放音频
                    st.audio(audio_bytes, format='audio/wav')
                    st.session_state.voice.append({"role": "assistant", "content": path})
       
        print(st.session_state.voice)
