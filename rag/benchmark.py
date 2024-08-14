import datetime
from lmdeploy import pipeline,TurbomindEngineConfig,GenerationConfig

backend_config = TurbomindEngineConfig(model_format="hf",cache_max_entry_count=0.01, session_len=32768)
gen_config = GenerationConfig(
        top_p=0.8,
        top_k=40,
        temperature=0,
        # max_new_tokens=4096
    )
pipe = pipeline("/group_share/merged", model_name="internlm2",backend_config=backend_config)
from build_prompt import build_wenlv_prompt
# pipe = pipeline("/group_share/merged",model_name="internlm2")

# warmup
inp = build_wenlv_prompt("你是谁？")
for i in range(1):
    print("Warm up...[{}/5]".format(i+1))
    response = pipe([inp])
    print(response[0].text)

# test speed
inp = "请介绍一下你自己。"
times = 10
total_words = 0
start_time = datetime.datetime.now()
for i in range(times):
    response = pipe([inp])
    total_words += len(response[0].text)
end_time = datetime.datetime.now()

delta_time = end_time - start_time
delta_time = delta_time.seconds + delta_time.microseconds / 1000000.0
speed = total_words / delta_time
print("Speed: {:.3f} words/s".format(speed))
