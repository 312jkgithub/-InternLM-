import base64
import requests
# CUDA_VISIBLE_DEVICES=0 python asr_server.py > log.txt 2>&1 &
import os
url = "http://127.0.0.1:7863/"

def asr_damo_api(text:str):
    headers = {'Content-Type': 'application/json'}
    print(text)
    data = {"text": text}
    response = requests.post(url+"tts", headers=headers, json=data)
    response = response.json()
    print(response)
    if response['code'] == 0:
        res = response['res']
        return res
    else:
        return response['msg']

if __name__ == '__main__':
    res = asr_damo_api("我叫琼琼，你叫什么")
    print(res)
