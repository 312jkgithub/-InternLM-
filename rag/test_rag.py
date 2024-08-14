from build_prompt import build_wenlv_prompt
# # from LLM import InternLM2Chat
from transformers import AutoTokenizer

# path="/root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b"
# model= InternLM2Chat(path)
prompt= build_wenlv_prompt(question="海南有什么好玩的")
print(prompt)
# chat=model.chat(prompt,[])
# lmdeploy serve api_server  /group_share/merged --server-port 7860   --model-name internlm2  --cache-max-entry-count 0.01  --api-keys lmdeploy

# print(chat[0])
from openai import OpenAI
client = OpenAI(
            api_key='lmdeploy',
            base_url="http://0.0.0.0:7860/v1"
        )
    
# print("*1000"*10)
# prompt= build_wenlv_prompt(question="你好")
tokenizer = AutoTokenizer.from_pretrained("/group_share/merged", trust_remote_code=True)
token=tokenizer.tokenize(prompt)
print(len(token))
model_name = client.models.list().data[0].id
# print( client.models.list().data[0])
response = client.chat.completions.create(model=model_name,
messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            top_p=0.8
        )
try:
    print(response)
    print("Completion Content:", response.choices[0].message.content)
except Exception as e:
    print("Error accessing response content:", e)
# print(response)
# print(response.choices[0].message.content)