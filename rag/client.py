from openai import OpenAI

# 启动服务命令为：
# CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server  /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b --server-port 7860 --api-keys lmdeploy
# lmdeploy serve api_server /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b  --server-port 7860


# lmdeploy serve api_server  /group_share/model/merged --server-port 7860  --cache-max-entry-count 0.01  --api-keys lmdeploy 

# 
from build_prompt import build_wenlv_prompt
from LLM import InternLM2Chat
import uvicorn
from fastapi import FastAPI
from utils import torch_gc
from lmdeploy import pipeline,TurbomindEngineConfig,GenerationConfig
import configparser
import os
# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件的目录路径（上一级）
parent_dir_path = os.path.dirname(current_file_path)
# 获取上两级目录的路径
grandparent_dir_path = os.path.dirname(parent_dir_path)

config = configparser.ConfigParser()
conf_path=grandparent_dir_path+'/config.ini'
print(conf_path)
config.read(conf_path)
app=app = FastAPI()
llm_model_path = config['paths']['llm_model_path']
# pip install autoawq
from pydantic import BaseModel
class RagItem(BaseModel):
    text : str # 输入
@app.post("/rag1")
def llm_response(question: RagItem):
    try:
        # path="/group_share/merged"
        assert llm_model_path != None
        model= InternLM2Chat(llm_model_path)
        prompt= build_wenlv_prompt(question=question.text)
        chat=model.chat(prompt,[])
        
        # return chat[0]
        torch_gc()
        result_dict = {"code": 0, "msg": "ok", "res": chat[0]}
        del model
    except Exception as e:
        result_dict = {"code": 1, "msg": str(e)}
    
    return result_dict



@app.post("/rag")
def response(question: RagItem):
    try:
        client = OpenAI(
            api_key='lmdeploy',
            base_url="http://0.0.0.0:7860/v1"
        )
        text = question.text
        print("*1000"*10)
        print(text)
        prompt= build_wenlv_prompt(question=text)
        print(prompt)
        model_name = client.models.list().data[0].id
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            top_p=0.8
        )
        print("*"*100)
        print(response.choices[0].message.content)
        result_dict = {"code": 0, "msg": "ok", "res": response.choices[0].message.content}
    except Exception as e:
        result_dict = {"code": 1, "msg": str(e)}
    return result_dict



@app.post("/rag_lmdeploy")
def rag_lmdeploy(question: RagItem):
    try:
     
        backend_config = TurbomindEngineConfig(model_format="hf", cache_max_entry_count=0.01,session_len=32768)
        # pipe = pipeline("/group_share/merged", model_name="internlm2",backend_config=backend_config)
        assert llm_model_path != None
        pipe = pipeline(llm_model_path, model_name="internlm2",backend_config=backend_config)
        prompt= build_wenlv_prompt(question=question.text)
        response = pipe([prompt])
        print(prompt)
        print(response[0].text)
        result_dict = {"code": 0, "msg": "ok", "res": response[0].text}
    except Exception as e:
        result_dict = {"code": 1, "msg": str(e)}
    
    return result_dict



if __name__ == '__main__':
    uvicorn.run("client:app", host='127.0.0.1', port=7861)
# print(llm_response("海南的美食"))