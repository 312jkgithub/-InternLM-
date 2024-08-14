
import torch
import torchaudio
from fastapi import FastAPI
# chat = ChatTTS.Chat()
import uvicorn
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import os, sys
sys.path.insert(0, os.path.abspath('third_party/Matcha-TTS'))

def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
MODEL_PATH="/root/wenlv/tts/model"

def count_files_in_folder():
    folder_path='/root/wenlv/audio/'
    count = 0
    # 遍历文件夹中所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        count += len(files)  # 累计文件数量
    return count


import random
from pydantic import BaseModel
app = FastAPI()
# 定义asr数据模型，用于接收POST请求中的数据
class TTSItem(BaseModel):
    text : str # 输入
@app.post("/tts")
def tts(item: TTSItem):
    cosyvoice = CosyVoice('/root/models/iic/CosyVoice-300M-SFT')
    # sft usage
    print(cosyvoice.list_avaliable_spks())
    output = cosyvoice.inference_sft(item.text, '中文女')
   
    # path="/root/wenlv/audio/output-01.wav"
    num = str(count_files_in_folder()+1)
    path= "/root/wenlv/audio/"+ num +".wav"
    torchaudio.save(path, output['tts_speech'], 22050)
    torch_gc()
   
    result_dict = {"code": 0, "msg": "ok", "res": path}

    return result_dict
if __name__ == '__main__':
    uvicorn.run("tts_server:app", host='127.0.0.1', port=7863)