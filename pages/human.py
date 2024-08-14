import  requests
import streamlit as st
# st.title("基于interlm的文旅小助手")

import base64
def render_video(video_path, width, height):
    """渲染视频文件的HTML标签"""
    # 读取视频文件
    with open(video_path, "rb") as video_file:
        video_data = video_file.read()
    # 将视频文件编码为Base64字符串
    video_base64 = base64.b64encode(video_data).decode()
    # HTML模板，使用了Base64编码的视频数据
    video_html = f"""
   <video width="{width}" height="{height}" controls>
       <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
       Your browser does not support the video tag.
   </video>
   """
    return video_html


# 使用Streamlit的自定义组件功能
def custom_video_component(video_path, width: int=160, height :int=160):
    return st.components.v1.html(render_video(video_path, width, height), height=height)

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


def hunman_api(path:str):
    headers = {'Content-Type': 'application/json'}
    print(path)
    data = {"path": path}
    url = "http://127.0.0.1:7864/"
    response = requests.post(url+"vidio", headers=headers, json=data)
    response = response.json()
    print(response)
    if response['code'] == 0:
        res = response['res']
        return res
    else:
        return response['msg']


    # st.header("列-1")
if "hunman" not in st.session_state.keys():
        st.session_state.hunman = []   

for hunman in st.session_state.hunman:
        with st.chat_message(hunman["role"]):
            if hunman["role"] == "user":
                 st.markdown(hunman["content"])
            else:
                path=hunman["content"]
                custom_video_component(path)
                # video_file = open(path, 'rb')
                # video_bytes = video_file.read()
            #本地视频
                # st.video(video_bytes,format="mp4",start_time=2)
if prompt := st.chat_input("请输入你的问题?"):
    # Display user message in chat message container
        with st.chat_message("user"):
             st.markdown(prompt)
    # Add user message to chat history
        st.session_state.hunman.append({"role": "user", "content": prompt})

        if st.session_state.hunman[-1]["role"] != "assistant":
            with st.spinner("Thinking..."):
                with st.chat_message("assistant"):
                    stream =rag_response(prompt)
                    path=tts_api(stream)
                    if os.path.exists(path):
                        print(f"File '{path}' exists.")
                    else:
                         print(f"File '{path}' does not exist.")
                    print(path)
                    file_path=path.strip('"')
                    hunman_path=hunman_api(file_path)
                    custom_video_component(hunman_path)
            #         video_file = open(hunman_path, 'rb')
            #         video_bytes = video_file.read()
            #         st.video(video_bytes,format="mp4",start_time=2)
                    st.session_state.hunman.append({"role": "assistant", "content": hunman_path})
       
        print(st.session_state.hunman)
